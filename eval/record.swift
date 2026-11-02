// Record one take on several Mac mics at once (48 kHz, 24-bit mono WAV). Press q or Enter to stop.
// Run through eval/record.sh. One AVCaptureSession per mic, writing straight to a file.
import AVFoundation

final class Done: NSObject, AVCaptureFileOutputRecordingDelegate {
    private let lock = NSLock()
    private var n = 0
    var finished: Int { lock.withLock { n } }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL, from connections: [AVCaptureConnection], error: Error?) {
        lock.withLock { n += 1 }
        let ok = (error as NSError?)?.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool ?? true
        if let error, !ok { print("\n\(url.lastPathComponent): \(error.localizedDescription)") }
    }
}

var args = Array(CommandLine.arguments.dropFirst())
guard !args.isEmpty else {
    print("usage: eval/record.sh <take-name> [mic name ...]")
    exit(1)
}
let take = args.removeFirst()
let mics = args.isEmpty ? ["MacBook Pro Microphone", "Studio Display XDR Microphone"] : args
let dir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("recordings")
try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
let available = AVCaptureDevice.DiscoverySession(deviceTypes: [.microphone, .external], mediaType: .audio,
                                                 position: .unspecified).devices
let done = Done()

var takes: [(name: String, session: AVCaptureSession, output: AVCaptureAudioFileOutput, url: URL)] = []
for mic in mics {
    guard let device = available.first(where: { $0.localizedName == mic }) else {
        print("mic not found: \(mic). Available: \(available.map(\.localizedName).joined(separator: ", "))")
        exit(1)
    }
    let session = AVCaptureSession()
    session.addInput(try AVCaptureDeviceInput(device: device))
    let output = AVCaptureAudioFileOutput()
    output.audioSettings = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48000, AVNumberOfChannelsKey: 1,
                            AVLinearPCMBitDepthKey: 24, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
                            AVLinearPCMIsNonInterleaved: false]
    session.addOutput(output)
    let slug = mic.lowercased().replacingOccurrences(of: " ", with: "-").replacingOccurrences(of: "-microphone", with: "")
    let url = dir.appendingPathComponent("\(take)_\(slug).wav")
    try? FileManager.default.removeItem(at: url)
    takes.append((mic.replacingOccurrences(of: " Microphone", with: ""), session, output, url))
}
for t in takes {
    t.session.startRunning()
    t.output.startRecording(to: t.url, outputFileType: .wav, recordingDelegate: done)
}

print("Recording '\(take)' — press q (or Enter) to stop")
let start = Date()
Thread.detachNewThread {
    while true {
        sleep(1)
        let levels = takes.map { t in
            String(format: "%@ %4.0f dB", t.name, t.output.connection(with: .audio)?.audioChannels.first?.averagePowerLevel ?? -120)
        }
        print(String(format: "\r  %3.0f s   ", Date().timeIntervalSince(start)) + levels.joined(separator: "   "), terminator: "")
        fflush(stdout)
    }
}
var saved = termios()
let isTTY = tcgetattr(0, &saved) == 0
if isTTY {
    var raw = saved
    raw.c_lflag &= ~tcflag_t(ICANON | ECHO)
    tcsetattr(0, TCSANOW, &raw)
}
var c: UInt8 = 0
while read(0, &c, 1) == 1, c != UInt8(ascii: "q"), c != 10 {}
if isTTY { tcsetattr(0, TCSANOW, &saved) }

for t in takes { t.output.stopRecording() }
let deadline = Date().addingTimeInterval(10)  // the WAV header is only written once each output reports finished
while done.finished < takes.count, Date() < deadline { RunLoop.main.run(until: Date().addingTimeInterval(0.05)) }
print("\nSaved:")
for t in takes {
    t.session.stopRunning()
    let secs = (try? AVAudioFile(forReading: t.url)).map { Double($0.length) / $0.fileFormat.sampleRate } ?? 0
    print(String(format: "  %@ (%.1f s)", t.url.lastPathComponent, secs))
}
