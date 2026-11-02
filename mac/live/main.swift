// Test tool for a CleanMyMic Core ML engine (from train/export.py; ours or DeepFilterNet3). The engine itself is
// Sources/CleanMyMic/Enhancer.swift, shared with the app.
//   mac/live.sh model.mlpackage                           live: default mic -> model -> output. USE HEADPHONES.
//                                                         Enter toggles bypass (A/B), q + Enter quits.
//   mac/live.sh model.mlpackage in.wav out.wav            file mode, 48 kHz mono in
//   mac/live.sh model.mlpackage --check-bypass            synthetic original-voice/resume regression; no mic
import AVFoundation

let args = CommandLine.arguments
let checkBypass = args.count == 3 && args[2] == "--check-bypass"
guard args.count == 2 || args.count == 4 || checkBypass else {
    print("usage: mac/live.sh model.mlpackage [in.wav out.wav | --check-bypass]")
    exit(1)
}
let enhancer = try Enhancer(URL(fileURLWithPath: args[1]))
enhancer.selfTest()

if checkBypass {
    let fresh = try Enhancer(URL(fileURLWithPath: args[1]))
    let oldVoice = (0..<hop).map { Float(sin(Double($0) * 0.04)) * 0.2 }
    for _ in 0..<40 { _ = try enhancer.process(oldVoice) }
    let savedState = enhancer.state.mapValues { a in (0..<a.count).map { a[$0].floatValue } }
    // A different signal reveals stale speech leaking back in when the model resumes.
    let voice = (0..<hop).map { Float(sin(Double($0) * 0.13)) * 0.1 }
    enhancer.mix = 0
    fresh.mix = 0
    for i in 0..<(enhancer.look + 8) {
        let raw = try enhancer.process(voice)
        _ = try fresh.process(voice)
        if i > enhancer.look { precondition(raw == voice, "Original voice was changed or silenced") }
    }
    precondition(savedState == enhancer.state.mapValues { a in (0..<a.count).map { a[$0].floatValue } },
                 "The model ran while noise removal was off")
    enhancer.mix = 1
    fresh.mix = 1
    for _ in 0..<8 {
        let resumed = try enhancer.process(voice), expected = try fresh.process(voice)
        precondition(zip(resumed, expected).allSatisfy { abs($0 - $1) < 1e-5 }, "Stale model state after bypass")
    }
    print("PASS: original voice is unchanged and cleanup resumes without stale speech")
    exit(0)
}

if args.count == 4 {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: args[2]))
    guard file.processingFormat.sampleRate == 48000, file.processingFormat.channelCount == 1 else {
        print("need 48 kHz mono input")
        exit(1)
    }
    let pcm = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: pcm)
    let samples = Array(UnsafeBufferPointer(start: pcm.floatChannelData![0], count: Int(pcm.frameLength)))
    var out: [Float] = []
    let t = Date()
    for i in stride(from: 0, through: samples.count - hop, by: hop) {
        out += try enhancer.process(Array(samples[i..<i + hop]))
    }
    let secs = Date().timeIntervalSince(t), dur = Double(samples.count) / 48000
    let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
    let outBuf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(out.count))!
    outBuf.frameLength = AVAudioFrameCount(out.count)
    out.withUnsafeBufferPointer { outBuf.floatChannelData![0].update(from: $0.baseAddress!, count: out.count) }
    try AVAudioFile(forWriting: URL(fileURLWithPath: args[3]), settings: format.settings).write(from: outBuf)
    print(String(format: "wrote %@ (%.1f s of audio in %.2f s, real-time factor %.3f)", args[3], dur, secs, secs / dur))
    exit(0)
}

let granted = await AVCaptureDevice.requestAccess(for: .audio)
guard granted else {
    print("microphone access denied (System Settings > Privacy & Security > Microphone)")
    exit(1)
}
let engine = AVAudioEngine()
let inFormat = engine.inputNode.outputFormat(forBus: 0)
let mono = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
let converter = AVAudioConverter(from: inFormat, to: mono)!  // any mic rate/channels -> 48 kHz mono
converter.downmix = true
let lock = NSLock()
var inFifo: [Float] = [], outFifo: [Float] = [], busy = 0.0, hops = 0
engine.inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(hop), format: inFormat) { buffer, _ in
    let converted = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity:
        AVAudioFrameCount(Double(buffer.frameLength) * 48000 / inFormat.sampleRate) + 64)!
    // Conversion calls this block synchronously; older SDKs incorrectly mark it @Sendable.
    nonisolated(unsafe) var fed = false
    converter.convert(to: converted, error: nil) { _, status in
        status.pointee = fed ? .noDataNow : .haveData
        defer { fed = true }
        return fed ? nil : buffer
    }
    inFifo += UnsafeBufferPointer(start: converted.floatChannelData![0], count: Int(converted.frameLength))
    while inFifo.count >= hop {
        let t = Date()
        let y = (try? enhancer.process(Array(inFifo.prefix(hop)))) ?? [Float](repeating: 0, count: hop)
        busy += Date().timeIntervalSince(t)
        hops += 1
        inFifo.removeFirst(hop)
        lock.lock()
        outFifo += y
        if outFifo.count > 9600 { outFifo.removeFirst(outFifo.count - 9600) }  // keep latency bounded
        lock.unlock()
    }
}
let source = AVAudioSourceNode(format: mono) { _, _, frameCount, abl in
    let out = UnsafeMutableAudioBufferListPointer(abl)[0].mData!.assumingMemoryBound(to: Float.self)
    lock.lock()
    let n = min(Int(frameCount), outFifo.count)
    for i in 0..<Int(frameCount) { out[i] = i < n ? outFifo[i] : 0 }
    outFifo.removeFirst(n)
    lock.unlock()
    return noErr
}
engine.attach(source)
engine.connect(source, to: engine.mainMixerNode, format: mono)
try engine.start()
print("live on \(AVCaptureDevice.default(for: .audio)?.localizedName ?? "default mic") (\(Int(inFormat.sampleRate)) Hz)."
      + " Enter = toggle bypass, q + Enter = quit")
while let line = readLine(), line != "q" {
    enhancer.mix = enhancer.mix == 0 ? 1 : 0
    print(enhancer.mix == 0 ? "BYPASS (raw mic)" : "MODEL ON",
          String(format: "| %.3f ms per 10 ms hop", hops > 0 ? busy / Double(hops) * 1000 : 0))
}
engine.stop()
