// The live path: mic -> 48 kHz mono frames -> engine (Enhancer) -> the chosen output device. One AVAudioEngine per
// device (they run on their own clocks). A sink node gets the mic's own small I/O buffers (an input tap would hand
// over ~100 ms at a time), a worker thread runs the model off the I/O threads, a source node plays what is ready.
//
// Devices change format under a running engine: AirPods switch to their call profile (mono, 16/24 kHz) the moment
// their mic opens, which stops an engine already playing to them. Each engine is re-wired on its own when that
// happens, so the mic is never closed and reopened (which would flip the AirPods back and forth).
import AVFoundation
import ObjCTry

nonisolated struct PlaybackWatch {
    private var lastSamples = 0
    private(set) var canRestart: Bool

    init(recovering: Bool) { canRestart = !recovering }

    mutating func advanced(to samples: Int) -> Bool {
        guard samples != lastSamples else { return false }
        lastSamples = samples
        canRestart = true
        return true
    }
}

/// Optional local raw+clean paired WAV recorder (worker thread only; off I/O threads; disabled unless `dir` is set).
/// Writes sample-aligned 48 kHz mono float32 WAVs (`*_raw.wav` and `*_clean.wav`) plus a JSON metadata sidecar.
nonisolated final class TakeRecorder {
    private let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
    private var rawFile: AVAudioFile?
    private var cleanFile: AVAudioFile?
    private var rawDelay: [[Float]]
    private let rawURL: URL
    private let cleanURL: URL
    private let metaURL: URL
    private var framesWritten = 0

    init?(dir: URL, micName: String, modelName: String, mix: Float, studioPolish: Bool, look: Int) {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            let df = DateFormatter()
            df.dateFormat = "yyyyMMdd-HHmmss"
            let ts = df.string(from: Date())
            let slug: (String) -> String = { s in
                let kept = s.lowercased().map { ($0.isLetter || $0.isNumber) ? $0 : "-" }
                return String(kept).replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            }
            let stem = "\(ts)_\(slug(micName))_\(slug(modelName))"
            rawURL = dir.appendingPathComponent("\(stem)_raw.wav")
            cleanURL = dir.appendingPathComponent("\(stem)_clean.wav")
            metaURL = dir.appendingPathComponent("\(stem).json")
            rawFile = try AVAudioFile(forWriting: rawURL, settings: format.settings,
                                      commonFormat: .pcmFormatFloat32, interleaved: false)
            cleanFile = try AVAudioFile(forWriting: cleanURL, settings: format.settings,
                                        commonFormat: .pcmFormatFloat32, interleaved: false)
            rawDelay = [[Float]](repeating: [Float](repeating: 0, count: hop), count: max(0, look) + 1)
            let meta: [String: Any] = [
                "startedAt": ISO8601DateFormatter().string(from: Date()),
                "mic": micName,
                "model": modelName,
                "mix": mix,
                "studioPolish": studioPolish,
                "sampleRate": 48000,
                "lookHops": look,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: metaURL)
            }
        } catch {
            return nil
        }
    }

    func append(raw: [Float], clean: [Float]) {
        guard let rawFile, let cleanFile, raw.count == clean.count, !raw.isEmpty else { return }
        rawDelay.append(raw)
        let alignedRaw = rawDelay.removeFirst()
        write(alignedRaw, to: rawFile)
        write(clean, to: cleanFile)
        framesWritten += 1
    }

    private func write(_ samples: [Float], to file: AVAudioFile) {
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress, let dst = buf.floatChannelData?[0] else { return }
            dst.update(from: base, count: samples.count)
        }
        try? file.write(from: buf)
    }

    func close() {
        rawFile = nil
        cleanFile = nil
        if framesWritten < 20 {  // drop accidental <0.2 s empty opens
            let fm = FileManager.default
            try? fm.removeItem(at: rawURL)
            try? fm.removeItem(at: cleanURL)
            try? fm.removeItem(at: metaURL)
        }
    }
}

nonisolated final class Pipeline: @unchecked Sendable {  // shared with audio threads: mutable state is behind `lock`
    private let mic = AVAudioEngine(), out = AVAudioEngine()
    private let input: AudioDeviceID, output: AudioDeviceID  // where each engine belongs; set again on every rewire
    private let mono = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
    private var sink: AVAudioSinkNode!, source: AVAudioSourceNode!
    private var format: AVAudioFormat?, converter: AVAudioConverter?  // the mic's; set only while `mic` is stopped
    private let enhancer: Enhancer  // the worker thread's alone once running
    private var drift = DriftResampler(targetFill: (3 * hop) / 2)  // worker thread only; absorbs mic/cable clock drift
    private let recorder: TakeRecorder?  // worker thread only; nil unless opt-in local take recording is enabled
    private let lock = NSLock()
    private var inFifo: [Float] = [], outFifo: [Float] = [], mix: Float, studioPolish: Bool, running = true, busy = 0.0, frames = 0
    private var delivered = 0
    var samplesDelivered: Int { lock.withLock { delivered } }
    private let wake = DispatchSemaphore(value: 0)
    private let rewiring = DispatchQueue(label: "CleanMyMic.rewire")  // device changes, off AVAudioEngine's threads
    private var observers: [NSObjectProtocol] = []
    private let onError: @Sendable (String) -> Void
    // ponytail: the two devices' clocks drift; this drops/pads samples at the bound. Resample adaptively if it clicks.
    private static let maxQueued = 3 * hop  // output at most ~30 ms ahead of the speaker

    /// onError: a device change the pipeline could not recover from (on some background thread).
    init(model: URL, input: AudioDeviceID, output: AudioDeviceID, mix: Float, studioPolish: Bool = true,
         saveTakesDir: URL? = nil, micName: String = "mic", modelName: String = "model",
         onError: @escaping @Sendable (String) -> Void) throws {
        let enhancer = try Enhancer(model)
        enhancer.mix = mix
        enhancer.preamp.enabled = true
        enhancer.polish = StudioPolish(enabled: studioPolish, limitPeaks: true)
        self.recorder = saveTakesDir.flatMap {
            TakeRecorder(dir: $0, micName: micName, modelName: modelName, mix: mix,
                         studioPolish: studioPolish, look: enhancer.look)
        }
        (self.enhancer, self.mix, self.studioPolish, self.onError) = (enhancer, mix, studioPolish, onError)
        (self.input, self.output) = (input, output)
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
    func setStudioPolish(_ enabled: Bool) { lock.withLock { studioPolish = enabled } }

    /// False once an engine has drifted to another device. Seen live: both fell back to the system defaults, so the
    /// app cleaned its own virtual mic and played into the headphones. Waits its turn behind any rewiring.
    var onIntendedDevices: Bool {
        rewiring.sync {
            mic.inputNode.audioUnit.flatMap(currentDevice) == input && out.outputNode.audioUnit.flatMap(currentDevice) == output
        }
    }

    /// Average model time per 10 ms frame, in ms.
    var msPerFrame: Double { lock.withLock { frames > 0 ? busy / Double(frames) * 1000 : 0 } }
    var framesDone: Int { lock.withLock { frames } }

    /// (Re)connects one engine for its device's current format and starts it.
    private func wire(mic isMic: Bool) throws {
        let engine = isMic ? mic : out
        engine.stop()
        // An engine that lost its device carries on with the system default unless told again. A device that is
        // gone fails here, and the controller starts over with whatever is connected now.
        guard let unit = isMic ? mic.inputNode.audioUnit : out.outputNode.audioUnit else { throw AudioError.noInput }
        try setDevice(unit, isMic ? input : output)
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
                onError("Lost the \(isMic ? "mic" : "output") when it changed format (\(error.localizedDescription)).")
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
            delivered += n  // silent audio counts, starvation padding does not
            outFifo.removeFirst(n)
        }
    }

    private func work() {
        defer { recorder?.close() }
        while true {
            wake.wait()
            while let (frame, mix, polishOn) = nextFrame() {
                enhancer.mix = mix
                enhancer.polish.enabled = polishOn
                let t = Date()
                let y = (try? enhancer.process(frame)) ?? frame
                let dt = Date().timeIntervalSince(t)
                recorder?.append(raw: frame, clean: y)
                let fill = lock.withLock { outFifo.count }
                let resampled = drift.process(y, currentFill: fill)
                lock.withLock {
                    busy += dt
                    frames += 1
                    outFifo += resampled
                    if outFifo.count > Self.maxQueued { outFifo.removeFirst(outFifo.count - Self.maxQueued) }
                }
            }
            if lock.withLock({ !running }) { return }
        }
    }

    private func nextFrame() -> ([Float], Float, Bool)? {
        lock.withLock {
            guard running, inFifo.count >= hop else { return nil }
            defer { inFifo.removeFirst(hop) }
            return (Array(inFifo.prefix(hop)), mix, studioPolish)
        }
    }
}

/// Runs body, turning an Objective-C exception from it into a thrown error.
private nonisolated func objc(_ body: () -> Void) throws {
    if let error = objcTry(body) { throw error }
}
