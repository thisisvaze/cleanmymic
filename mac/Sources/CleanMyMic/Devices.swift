// Core Audio devices: physical microphones and the hidden output feeding our virtual microphone.
import AudioToolbox
import CoreAudio
import Foundation

nonisolated struct AudioDevice: Identifiable, Hashable, Sendable {
    let id: AudioDeviceID
    let uid: String  // stable across reboots, unlike id: what settings store
    let name: String
    let inputs: Bool, outputs: Bool
    var isVirtual = false  // software audio (Zoom's and Teams' devices, loopback cables), not a microphone

    var isMicrophoneSource: Bool {
        inputs && !isVirtual && uid != micUID && uid != cableUID && !name.localizedCaseInsensitiveContains("BlackHole")
            && !uid.hasPrefix("CADefaultDeviceAggregate")  // Core Audio's private helper for our own audio engine
    }
}

/// Our virtual mic (driver/): apps see the mic, and only we write to the hidden cable behind it.
nonisolated let micUID = "CleanMyMic_UID", cableUID = "CleanMyMic_2_UID"

nonisolated func microphoneSource(_ uids: [String?], in devices: [AudioDevice]) -> AudioDeviceID? {
    let real = devices.filter(\.isMicrophoneSource)
    for uid in uids { if let d = real.first(where: { $0.uid == uid }) { return d.id } }
    return (real.first { isBuiltIn($0.id) } ?? real.first)?.id
}

nonisolated enum SystemInputChange: Equatable {
    case none, releaseOwnership
    case claim(AudioDeviceID, previousUID: String), restore(AudioDeviceID)

    static func decide(current: AudioDevice?, devices: [AudioDevice], previousUID: String?,
                       preferredInputUID: String? = nil, enabled: Bool, allowClaim: Bool,
                       leaveVirtualInput: Bool = false) -> Self {
        guard let current else { return .none }  // Core Audio may be between devices
        // Pick the best non-CleanMyMic input to hand back to macOS: explicit user choice in CleanMyMic (if connected),
        // then whatever macOS was using before CleanMyMic claimed it (even a third-party virtual input), then built-in.
        func restoreTarget() -> AudioDeviceID? {
            if let preferredInputUID, !preferredInputUID.isEmpty, preferredInputUID != micUID,
               let chosen = devices.first(where: { $0.uid == preferredInputUID && $0.isMicrophoneSource }) {
                return chosen.id
            }
            if let previousUID,
               let original = devices.first(where: { $0.uid == previousUID && $0.inputs && $0.uid != micUID }) {
                return original.id
            }
            return microphoneSource([], in: devices)
        }
        if previousUID != nil {
            guard current.uid == micUID else { return .releaseOwnership }
            guard !enabled else { return .none }
            guard let target = restoreTarget() else { return .none }
            return .restore(target)
        }
        // Even if CleanMyMic was chosen manually in macOS Sound settings, turning CleanMyMic off or quitting
        // always switches macOS Sound away so the Mac is never left on an off or dead virtual mic.
        if leaveVirtualInput, current.uid == micUID {
            guard let target = restoreTarget() else { return .none }
            return .restore(target)
        }
        guard enabled, allowClaim, current.uid != micUID, !current.uid.isEmpty,
              cleanMicOutput(in: devices) != nil,
              let mic = devices.first(where: { $0.uid == micUID && $0.inputs }) else { return .none }
        return .claim(mic.id, previousUID: current.uid)
    }
}

nonisolated func cleanMicOutput(in devices: [AudioDevice]) -> AudioDevice? {
    guard devices.contains(where: { $0.uid == micUID && $0.inputs }) else { return nil }
    return devices.first { $0.uid == cableUID && $0.outputs }
}

nonisolated func deviceIsRunning(_ device: AudioDeviceID) throws -> Bool {
    var property = address(kAudioDevicePropertyDeviceIsRunningSomewhere)
    var running: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let result = AudioObjectGetPropertyData(device, &property, 0, nil, &size, &running)
    guard result == noErr else { throw AudioError.activity(result) }
    return running != 0
}

/// True when this app is the only one recording from a device. Our own engine has ended up reading the virtual mic
/// (it fell back to the system's default input), and counting that kept the app "in use" on its own silence.
/// `readers`: each process that is recording, with the UIDs it records from.
nonisolated func onlySelfRecords(from uid: String, readers: [(pid: pid_t, uids: [String])], me: pid_t) -> Bool {
    guard readers.contains(where: { $0.pid == me && $0.uids.contains(uid) }) else { return false }
    // Another app can reach a mic through Core Audio's private aggregate, so one of those counts as a possible reader.
    return !readers.contains { $0.pid != me && $0.uids.contains { $0 == uid || $0.hasPrefix("CADefaultDeviceAggregate") } }
}

/// Whether another app is recording from a device. Falls back to "running somewhere" if readers cannot be listed.
nonisolated func otherAppIsRecording(from device: AudioDeviceID) throws -> Bool {
    guard try deviceIsRunning(device) else { return false }
    let system = AudioObjectID(kAudioObjectSystemObject)
    var readers: [(pid: pid_t, uids: [String])] = []
    for process in objects(system, kAudioHardwarePropertyProcessObjectList) {
        var addr = address(kAudioProcessPropertyIsRunningInput)
        var recording = UInt32(0), pid = pid_t(0)
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(process, &addr, 0, nil, &size, &recording) == noErr, recording != 0 else { continue }
        addr = address(kAudioProcessPropertyPID)
        size = UInt32(MemoryLayout<pid_t>.size)
        guard AudioObjectGetPropertyData(process, &addr, 0, nil, &size, &pid) == noErr else { continue }
        let inputs = objects(process, kAudioProcessPropertyDevices, kAudioObjectPropertyScopeInput)
        readers.append((pid, inputs.map { string($0, kAudioDevicePropertyDeviceUID) }))
    }
    return !onlySelfRecords(from: string(device, kAudioDevicePropertyDeviceUID), readers: readers, me: getpid())
}

private nonisolated func objects(_ owner: AudioObjectID, _ selector: AudioObjectPropertySelector,
                                 _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> [AudioObjectID] {
    var addr = address(selector, scope)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(owner, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(owner, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

nonisolated func audioDevices() -> [AudioDevice] {
    var size: UInt32 = 0
    var addr = address(kAudioHardwarePropertyDevices)
    let system = AudioObjectID(kAudioObjectSystemObject)
    guard AudioObjectGetPropertyDataSize(system, &addr, 0, nil, &size) == noErr else { return [] }
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(system, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    let cable = hiddenDevice(uid: cableUID)  // hidden devices aren't listed
    return (ids + (cable.map { [$0] } ?? [])).map {
        AudioDevice(id: $0, uid: string($0, kAudioDevicePropertyDeviceUID),
                    name: $0 == cable ? "CleanMyMic (for calls)" : string($0, kAudioObjectPropertyName),
                    inputs: channels($0, kAudioObjectPropertyScopeInput) > 0,
                    outputs: channels($0, kAudioObjectPropertyScopeOutput) > 0,
                    isVirtual: transport($0) == kAudioDeviceTransportTypeVirtual)
    }
}

private nonisolated func hiddenDevice(uid: String) -> AudioDeviceID? {
    var addr = address(kAudioHardwarePropertyTranslateUIDToDevice)
    var uid = uid as CFString
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let ok = withUnsafePointer(to: &uid) {
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr,
                                   UInt32(MemoryLayout<CFString>.size), $0, &size, &id) == noErr
    }
    return ok && id != 0 ? id : nil
}

nonisolated func isBuiltIn(_ id: AudioDeviceID) -> Bool { transport(id) == kAudioDeviceTransportTypeBuiltIn }

/// How long to wait between opening a mic and starting the output. A Bluetooth headset switches to its call
/// profile once its mic opens, which stops an engine already playing to it; no other mic changes anything.
nonisolated func outputDelay(micTransport: UInt32?) -> TimeInterval {
    micTransport == kAudioDeviceTransportTypeBluetooth || micTransport == kAudioDeviceTransportTypeBluetoothLE ? 0.3 : 0
}

nonisolated func outputDelay(afterOpening mic: AudioDeviceID) -> TimeInterval { outputDelay(micTransport: transport(mic)) }

private nonisolated func transport(_ id: AudioDeviceID) -> UInt32? {
    var addr = address(kAudioDevicePropertyTransportType)
    var transport = UInt32(0)
    var size = UInt32(MemoryLayout<UInt32>.size)
    return AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &transport) == noErr ? transport : nil
}

/// The Mac's own speakers: a headphone test through them feeds the voice back into the mic.
// ponytail: only built-in speakers are detected; external speakers look like headphones to Core Audio.
nonisolated func isBuiltInSpeaker(_ id: AudioDeviceID) -> Bool {
    var addr = address(kAudioDevicePropertyDataSource, kAudioObjectPropertyScopeOutput)
    var source = UInt32(0)
    var size = UInt32(MemoryLayout<UInt32>.size)
    let ok = AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &source) == noErr
    return isBuiltIn(id) && ok && source == 0x6973_706B  // 'ispk', internal speaker
}

/// Why "Mic check" can't start now, or nil.
nonisolated func headphoneTestBlocker(output: AudioDevice?, isSpeaker: Bool, inCall: Bool) -> String? {
    if inCall { return "Unavailable while an app uses CleanMyMic" }
    guard let output, output.outputs, output.uid != cableUID else { return "No headphones found" }
    return isSpeaker ? "Connect headphones: speakers would echo" : nil
}

/// Calls action, on the main queue, whenever a property of the audio system changes.
nonisolated func watch(_ selector: AudioObjectPropertySelector, _ action: @escaping @Sendable () -> Void) {
    var addr = address(selector)
    AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, .main) { _, _ in action() }
}

/// The system's default mic or output.
nonisolated func defaultDevice(input: Bool) -> AudioDeviceID? {
    var addr = address(input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice)
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let ok = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id) == noErr
    return ok && id != 0 ? id : nil
}

/// Makes a device the system's default mic: what apps that follow the system use.
nonisolated func setDefaultInput(_ id: AudioDeviceID) throws {
    var addr = address(kAudioHardwarePropertyDefaultInputDevice)
    var id = id
    let result = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil,
                               UInt32(MemoryLayout<AudioDeviceID>.size), &id)
    guard result == noErr else { throw AudioError.defaultInput }
}

/// Points an AVAudioEngine input or output node (an AUHAL unit) at a device; before the engine starts.
nonisolated func setDevice(_ unit: AudioUnit, _ id: AudioDeviceID) throws {
    var id = id
    let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &id,
                                      UInt32(MemoryLayout<AudioDeviceID>.size))
    guard status == noErr else { throw AudioError.device(status) }
}

/// The device an engine's I/O unit is on now.
nonisolated func currentDevice(_ unit: AudioUnit) -> AudioDeviceID? {
    var id = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let status = AudioUnitGetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &id, &size)
    return status == noErr && id != 0 ? id : nil
}

/// The audio path has hung: starting for too long, or running with no health check getting through.
/// Seen live with AirPods changing mode: a recovery blocked inside AVAudioEngine and every restart queued behind it.
nonisolated func audioIsStuck(startingSince: Date?, running: Bool, lastHealthy: Date, now: Date, limit: TimeInterval = 15) -> Bool {
    if let startingSince { return now.timeIntervalSince(startingSince) > limit }
    return running && now.timeIntervalSince(lastHealthy) > limit
}

nonisolated enum AudioError: Error, CustomStringConvertible {
    case device(OSStatus), activity(OSStatus), defaultInput, noInput, noModel(String)
    var description: String {
        switch self {
        case .device(let s): "can't use that device (Core Audio error \(s))"
        case .activity(let s): "Unable to monitor microphone use (Core Audio error \(s))."
        case .defaultInput: "Could not change the Mac’s input. Check Sound settings."
        case .noInput: "that mic gives no audio"
        case .noModel(let n): "the \(n) model is missing from the app"
        }
    }
}

nonisolated func address(_ selector: AudioObjectPropertySelector,
                                 _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

private nonisolated func string(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String {
    var addr = address(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &value) == noErr, let value else { return "" }
    return value.takeRetainedValue() as String
}

private nonisolated func channels(_ id: AudioDeviceID, _ scope: AudioObjectPropertyScope) -> Int {
    var addr = address(kAudioDevicePropertyStreamConfiguration, scope)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let list = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { list.deallocate() }
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, list) == noErr else { return 0 }
    return UnsafeMutableAudioBufferListPointer(list.assumingMemoryBound(to: AudioBufferList.self)).reduce(0) {
        $0 + Int($1.mNumberChannels)
    }
}
