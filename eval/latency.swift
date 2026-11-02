// File-only processing benchmark; no microphone capture. Compile with the app's Enhancer.swift.
import AVFoundation
import Foundation

@main struct FrameBenchmark {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            print("usage: coreml-latency model.mlmodelc input.wav [more.wav ...]")
            exit(1)
        }
        let model = URL(fileURLWithPath: args[1])
        var rows = [[String: Any]]()
        for path in args.dropFirst(2) {
            let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
            guard file.processingFormat.sampleRate == 48000 && file.processingFormat.channelCount == 1 else {
                print("need 48 kHz mono input: \(path)")
                exit(1)
            }
            guard file.length >= AVAudioFramePosition(hop) else {
                print("need at least one 10 ms frame: \(path)")
                exit(1)
            }
            let pcm = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
            try file.read(into: pcm)
            let count = Int(pcm.frameLength)
            guard count >= hop else {
                print("need at least one 10 ms frame: \(path)")
                exit(1)
            }
            let samples = Array(UnsafeBufferPointer(start: pcm.floatChannelData![0], count: count))
            let initStart = DispatchTime.now().uptimeNanoseconds
            let enhancer = try Enhancer(model)
            let initMs = Double(DispatchTime.now().uptimeNanoseconds - initStart) / 1e6
            var times = [Double]()
            var checksum: Double = 0
            for i in stride(from: 0, through: count - hop, by: hop) {
                let frame = Array(samples[i..<i + hop])
                let start = DispatchTime.now().uptimeNanoseconds
                let output = try enhancer.process(frame)
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                precondition(output.count == hop && output.allSatisfy(\.isFinite))
                checksum += output.reduce(0) { $0 + Double($1 * $1) }
            }
            let sorted = times.sorted(), mean = times.reduce(0, +) / Double(times.count)
            func percentile(_ p: Double) -> Double { sorted[Int(ceil(p * Double(sorted.count))) - 1] }
            rows.append(["input": path, "frames": times.count, "audio_seconds": Double(times.count) / 100,
                         "mean_ms": mean, "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
                         "p99_ms": percentile(0.99), "max_ms": sorted.last!, "frames_over_10ms": times.filter { $0 > 10 }.count,
                         "rtf": mean / 10, "initialization_ms": initMs, "lookahead_hops": enhancer.look,
                         "window_ms": Double(win) / 48, "hop_ms": Double(hop) / 48,
                         "signal_shift_ms": Double(1 + enhancer.look) * 10,
                         "algorithmic_window_plus_lookahead_ms": Double(win) / 48 + Double(enhancer.look) * 10,
                         "output_energy_checksum": checksum])
        }
        let result: [String: Any] = ["model": model.path, "date": ISO8601DateFormatter().string(from: Date()),
            "method": "Single sequential unpaced full-strength stream; app Enhancer.process timed per frame; Core ML default compute units; model load and warm-up excluded; input/output file IO excluded; no hardware or virtual-device routing; other system workloads not controlled; first processed frame included; no outputs saved.", "clips": rows]
        let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        print(String(data: data, encoding: .utf8)!)
    }
}
