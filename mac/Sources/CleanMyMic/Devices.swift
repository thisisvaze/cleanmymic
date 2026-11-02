// Core Audio devices: physical microphones and the hidden output feeding our virtual microphone.
import AudioToolbox
import CoreAudio

nonisolated struct AudioDevice: Identifiable, Hashable, Sendable {
    let id: AudioDeviceID
    let uid: String  // stable across reboots, unlike id: what settings store
    let name: String
    let inputs: Bool, outputs: Bool

    var isMicrophoneSource: Bool {
        inputs && uid != micUID && uid != cableUID && !name.localizedCaseInsensitiveContains("BlackHole")
    }
}

/// Our virtual mic (driver/): apps see the mic, and only we write to the hidden cable behind it.
nonisolated let micUID = "CleanMyMic_UID", cableUID = "CleanMyMic_2_UID"

nonisolated func microphoneSource(_ uids: [String?], in devices: [AudioDevice]) -> AudioDeviceID? {
    let real = devices.filter(\.isMicrophoneSource)
    for uid in uids { if let d = real.first(where: { $0.uid == uid }) { return d.id } }
    return (real.first { isBuiltIn($0.id) } ?? real.first)?.id
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
                    outputs: channels($0, kAudioObjectPropertyScopeOutput) > 0)
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

nonisolated func isBuiltIn(_ id: AudioDeviceID) -> Bool {
    var addr = address(kAudioDevicePropertyTransportType)
    var transport = UInt32(0)
    var size = UInt32(MemoryLayout<UInt32>.size)
    let ok = AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &transport) == noErr
    return ok && transport == kAudioDeviceTransportTypeBuiltIn
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
    guard result == noErr, defaultDevice(input: true) == id else { throw AudioError.defaultInput }
}

/// Points an AVAudioEngine input or output node (an AUHAL unit) at a device; before the engine starts.
nonisolated func setDevice(_ unit: AudioUnit, _ id: AudioDeviceID) throws {
    var id = id
    let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &id,
                                      UInt32(MemoryLayout<AudioDeviceID>.size))
    guard status == noErr else { throw AudioError.device(status) }
}

nonisolated enum AudioError: Error, CustomStringConvertible {
    case device(OSStatus), activity(OSStatus), defaultInput, noInput, noModel(String)
    var description: String {
        switch self {
        case .device(let s): "can't use that device (Core Audio error \(s))"
        case .activity(let s): "Unable to monitor microphone use (Core Audio error \(s))."
        case .defaultInput: "Choose CleanMyMic in System Settings > Sound > Input to finish setup."
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
