// The live path: mic -> 48 kHz mono frames -> engine (Enhancer) -> the chosen output device. One AVAudioEngine per
// device (they run on their own clocks). A sink node gets the mic's own small I/O buffers (an input tap would hand
// over ~100 ms at a time), a worker thread runs the model off the I/O threads, a source node plays what is ready.
//
// Devices change format under a running engine: AirPods switch to their call profile (mono, 16/24 kHz) the moment
// their mic opens, which stops an engine already playing to them. Each engine is re-wired on its own when that
// happens, so the mic is never closed and reopened (which would flip the AirPods back and forth).
import AVFoundation
import ObjCTry

nonisolated final class Pipeline: @unchecked Sendable {  // shared with audio threads: mutable state is behind `lock`
    private let mic = AVAudioEngine(), out = AVAudioEngine()
    private let mono = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
    private var sink: AVAudioSinkNode!, source: AVAudioSourceNode!
    private var format: AVAudioFormat?, converter: AVAudioConverter?  // the mic's; set only while `mic` is stopped
    private let enhancer: Enhancer  // the worker thread's alone once running
    private let lock = NSLock()
    private var inFifo: [Float] = [], outFifo: [Float] = [], mix: Float, running = true, busy = 0.0, frames = 0
    private let wake = DispatchSemaphore(value: 0)
    private let rewiring = DispatchQueue(label: "CleanMyMic.rewire")  // device changes, off AVAudioEngine's threads
    private var observers: [NSObjectProtocol] = []
    private let onError: @Sendable (String) -> Void
    // ponytail: the two devices' clocks drift; this drops/pads samples at the bound. Resample adaptively if it clicks.
    private static let maxQueued = 3 * hop  // output at most ~30 ms ahead of the speaker

    /// onError: a device change the pipeline could not recover from (on some background thread).
    init(model: URL, input: AudioDeviceID, output: AudioDeviceID, mix: Float,
         onError: @escaping @Sendable (String) -> Void) throws {
        let enhancer = try Enhancer(model)
        enhancer.mix = mix
        (self.enhancer, self.mix, self.onError) = (enhancer, mix, onError)
        sink = AVAudioSinkNode { [unowned self] _, count, list in
            receive(list, count)
            return noErr
        }
        source = AVAudioSourceNode(format: mono) { [unowned self] _, _, count, list in
            render(list, Int(count))
            return noErr
        }
        mic.attach(sink)
        out.attach(source)
        try setDevice(mic.inputNode.audioUnit!, input)
        try setDevice(out.outputNode.audioUnit!, output)
        for isMic in [true, false] {
            observers.append(NotificationCenter.default.addObserver(
                forName: .AVAudioEngineConfigurationChange, object: isMic ? mic : out, queue: nil) { [weak self] _ in
                // let the device settle first; the pipeline may be gone by then
                self?.rewiring.asyncAfter(deadline: .now() + 0.3) { [weak self] in self?.recover(isMic) }
            })
        }
    }

    func start() throws {
        do {
            try Task.checkCancellation()  // the caller may have closed its mic while Core ML loaded
            let worker = Thread { self.work() }  // holds the pipeline until it sees `running` go false
            worker.qualityOfService = .userInteractive
            worker.start()
            try rewiring.sync {  // wiring, stopping and recovering run one at a time on this queue
                try retrying { try wire(mic: true) }
                // a Bluetooth headset switches to its call profile once its mic opens: wire the output after that
                // ponytail: a fixed wait; a later switch is still caught by the configuration-change handler
                Thread.sleep(forTimeInterval: 0.3)
                try Task.checkCancellation()
                try retrying { try wire(mic: false) }
            }
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        lock.withLock { running = false }
        rewiring.sync {  // after any recovery in flight, which could otherwise restart an engine stopped here
            for o in observers { NotificationCenter.default.removeObserver(o) }
            mic.stop()
            out.stop()
        }
        wake.signal()
    }

    func setMix(_ m: Float) { lock.withLock { mix = m } }

    /// Average model time per 10 ms frame, in ms.
    var msPerFrame: Double { lock.withLock { frames > 0 ? busy / Double(frames) * 1000 : 0 } }
    var framesDone: Int { lock.withLock { frames } }

    /// (Re)connects one engine for its device's current format and starts it.
    private func wire(mic isMic: Bool) throws {
        let engine = isMic ? mic : out
        engine.stop()
        if isMic {
            let f = mic.inputNode.inputFormat(forBus: 0)  // the device's format now; outputFormat can be stale after a change
            guard f.sampleRate > 0, f.channelCount > 0 else { throw AudioError.noInput }
            format = f
            converter = f.sampleRate == 48000 ? nil : AVAudioConverter(from: f, to: mono)
            converter?.downmix = true
            try objc { mic.connect(mic.inputNode, to: sink, format: f) }
        } else {
            try objc { out.connect(source, to: out.mainMixerNode, format: mono) }
        }
        var started: Error?
        try objc {
            do { try engine.start() } catch { started = error }
        }
        if let started { throw started }
    }

    /// A device changed format under a running engine (which stopped itself): re-wire it. On `rewiring`.
    private func recover(_ isMic: Bool) {
        guard lock.withLock({ running }) else { return }
        do {
            try retrying(6) { try wire(mic: isMic) }
        } catch {
            if lock.withLock({ running }) {
                onError("Lost the \(isMic ? "mic" : "output") when it changed format (\(error.localizedDescription))."
                        + " Quit and reopen CleanMyMic.")
            }
        }
    }

    /// A device mid-change can refuse a format for a moment: retries body while the pipeline is running.
    private func retrying(_ attempts: Int = 4, _ body: () throws -> Void) throws {
        for attempt in 1...attempts {
            do {
                return try body()
            } catch where attempt < attempts && lock.withLock({ running }) {
                Thread.sleep(forTimeInterval: 0.3)
            }
        }
    }

    /// Mic I/O thread: its buffer as 48 kHz mono, queued for the worker.
    private func receive(_ list: UnsafePointer<AudioBufferList>, _ count: AVAudioFrameCount) {
        guard let format else { return }
        var samples: [Float]
        if let converter, let input = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: list, deallocator: nil) {
            // ponytail: allocates on the I/O thread for mics not at 48 kHz (AirPods); preallocate if it glitches
            let converted = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity:
                AVAudioFrameCount(Double(count) * 48000 / format.sampleRate) + 64)!
            // Conversion calls this block synchronously; older SDKs incorrectly mark it @Sendable.
            nonisolated(unsafe) var fed = false
            converter.convert(to: converted, error: nil) { _, status in
                status.pointee = fed ? .noDataNow : .haveData
                defer { fed = true }
                return fed ? nil : input
            }
            samples = Array(UnsafeBufferPointer(start: converted.floatChannelData![0], count: Int(converted.frameLength)))
        } else {
            let b = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))[0]
            let p = b.mData!.assumingMemoryBound(to: Float.self), n = Int(b.mNumberChannels)  // n > 1: interleaved
            samples = (0..<Int(count)).map { p[$0 * n] }  // the first channel
        }
        lock.withLock { inFifo += samples }
        wake.signal()
    }

    /// Output I/O thread: what is ready, silence if nothing is.
    private func render(_ list: UnsafeMutablePointer<AudioBufferList>, _ count: Int) {
        let out = UnsafeMutableAudioBufferListPointer(list)[0].mData!.assumingMemoryBound(to: Float.self)
        lock.withLock {
            let n = min(count, outFifo.count)
            for i in 0..<count { out[i] = i < n ? outFifo[i] : 0 }
            outFifo.removeFirst(n)
        }
    }

    private func work() {
        while true {
            wake.wait()
            while let (frame, mix) = nextFrame() {
                enhancer.mix = mix
                let t = Date()
                let y = (try? enhancer.process(frame)) ?? frame
                let dt = Date().timeIntervalSince(t)
                lock.withLock {
                    busy += dt
                    frames += 1
                    outFifo += y
                    if outFifo.count > Self.maxQueued { outFifo.removeFirst(outFifo.count - Self.maxQueued) }
                }
            }
            if lock.withLock({ !running }) { return }
        }
    }

    private func nextFrame() -> ([Float], Float)? {
        lock.withLock {
            guard running, inFifo.count >= hop else { return nil }
            defer { inFifo.removeFirst(hop) }
            return (Array(inFifo.prefix(hop)), mix)
        }
    }
}

/// Runs body, turning an Objective-C exception from it into a thrown error.
private nonisolated func objc(_ body: () -> Void) throws {
    if let error = objcTry(body) { throw error }
}
