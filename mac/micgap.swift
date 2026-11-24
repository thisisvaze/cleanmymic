// How long does the CleanMyMic virtual microphone stay silent after an app opens it? Whatever is said in that gap
// is lost, so this is the number behind "the first word was cut off". Needs the app running and no call in progress.
//   swift mac/micgap.swift [opens]      default 3
import CoreAudio
import Foundation

func address(_ selector: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func device(uid: String) -> AudioDeviceID? {
    var addr = address(kAudioHardwarePropertyTranslateUIDToDevice)
    var uid = uid as CFString
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let ok = withUnsafePointer(to: &uid) {
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, UInt32(MemoryLayout<CFString>.size), $0, &size, &id) == noErr
    }
    return ok && id != 0 ? id : nil
}

func isRunning(_ id: AudioDeviceID) -> Bool {
    var addr = address(kAudioDevicePropertyDeviceIsRunningSomewhere)
    var running: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    return AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &running) == noErr && running != 0
}

final class Gap: @unchecked Sendable {  // written on the I/O thread, read after the device stops
    var firstCallback: Date?, firstSound: Date?
}

guard let mic = device(uid: "CleanMyMic_UID") else { print("CleanMyMic virtual microphone not found"); exit(1) }
let trials = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1]) ?? 3 : 3
var results: [Double] = []
for trial in 1...trials {
    if isRunning(mic) { print("trial \(trial): the virtual mic is already in use by another app; not measured"); exit(2) }
    let gap = Gap()
    var proc: AudioDeviceIOProcID?
    let status = AudioDeviceCreateIOProcIDWithBlock(&proc, mic, nil) { _, input, _, _, _ in
        if gap.firstCallback == nil { gap.firstCallback = Date() }
        guard gap.firstSound == nil else { return }
        for buffer in UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input)) {
            guard let data = buffer.mData else { continue }
            let samples = UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float32.self), count: Int(buffer.mDataByteSize) / 4)
            if samples.contains(where: { $0 != 0 }) { gap.firstSound = Date(); return }
        }
    }
    guard status == noErr, let proc else { print("could not open the virtual mic (\(status))"); exit(1) }
    let opened = Date()
    AudioDeviceStart(mic, proc)
    let deadline = Date().addingTimeInterval(6)
    while gap.firstSound == nil && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
    AudioDeviceStop(mic, proc)
    AudioDeviceDestroyIOProcID(mic, proc)
    let ms = { (d: Date?) in d.map { String(format: "%.0f ms", $0.timeIntervalSince(opened) * 1000) } ?? "never" }
    print("trial \(trial): first buffer after \(ms(gap.firstCallback)), first sound after \(ms(gap.firstSound))")
    if let s = gap.firstSound { results.append(s.timeIntervalSince(opened) * 1000) }
    Thread.sleep(forTimeInterval: 5)  // let the app notice the mic closed and go back to Ready
}
if results.isEmpty {
    print("no sound arrived: is CleanMyMic running, and may this terminal use the microphone?")
    exit(3)
}
print(String(format: "silent for %.0f ms on average (%.0f to %.0f) over %d opens", results.reduce(0, +) / Double(results.count), results.min()!, results.max()!, results.count))
