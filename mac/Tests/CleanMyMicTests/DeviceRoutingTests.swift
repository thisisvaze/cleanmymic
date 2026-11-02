import Testing
@testable import CleanMyMic

@Test func onlyThePairedVirtualOutputCanReceiveCleanAudio() {
    let mic = AudioDevice(id: 1, uid: micUID, name: "CleanMyMic", inputs: true, outputs: false)
    let cable = AudioDevice(id: 2, uid: cableUID, name: "Hidden cable", inputs: false, outputs: true)
    let headphones = AudioDevice(id: 3, uid: "headphones", name: "Headphones", inputs: false, outputs: true)
    let impostor = AudioDevice(id: 4, uid: "other", name: "CleanMyMic", inputs: true, outputs: true)
    #expect(cleanMicOutput(in: [headphones, impostor, mic, cable]) == cable)
    #expect(cleanMicOutput(in: [headphones, impostor, mic]) == nil)
    #expect(cleanMicOutput(in: [headphones, impostor, cable]) == nil)
    #expect(cleanMicOutput(in: [mic, AudioDevice(id: 2, uid: cableUID, name: "Wrong direction",
                                               inputs: true, outputs: false)]) == nil)
}

@Test func headphoneTestNeedsHeadphonesAndNoCall() {
    let headphones = AudioDevice(id: 3, uid: "airpods", name: "AirPods", inputs: true, outputs: true)
    let cable = AudioDevice(id: 2, uid: cableUID, name: "Hidden cable", inputs: false, outputs: true)
    #expect(headphoneTestBlocker(output: headphones, isSpeaker: false, inCall: false) == nil)
    #expect(headphoneTestBlocker(output: headphones, isSpeaker: true, inCall: false) != nil)  // feedback
    #expect(headphoneTestBlocker(output: headphones, isSpeaker: false, inCall: true) != nil)  // the call keeps the mic
    #expect(headphoneTestBlocker(output: cable, isSpeaker: false, inCall: false) != nil)  // would reach the call app
    #expect(headphoneTestBlocker(output: nil, isSpeaker: false, inCall: false) != nil)
}

@Test func microphoneChoicesExcludeLoopbackAndOutputDevices() {
    let devices = [
        AudioDevice(id: 1, uid: "physical", name: "Mac microphone", inputs: true, outputs: false),
        AudioDevice(id: 2, uid: micUID, name: "CleanMyMic", inputs: true, outputs: false),
        AudioDevice(id: 3, uid: cableUID, name: "Cable", inputs: true, outputs: true),
        AudioDevice(id: 4, uid: "blackhole", name: "BlackHole 2ch", inputs: true, outputs: true),
        AudioDevice(id: 5, uid: "speakers", name: "Speakers", inputs: false, outputs: true),
        AudioDevice(id: 6, uid: "CADefaultDeviceAggregate-123-1", name: "CADefaultDeviceAggregate-123-1",
                    inputs: true, outputs: true),
        AudioDevice(id: 7, uid: "zoom", name: "ZoomAudioDevice", inputs: true, outputs: true, isVirtual: true),
    ]
    #expect(devices.filter(\.isMicrophoneSource).map(\.uid) == ["physical"])
}

@Test func automaticMicrophoneRemembersThePhysicalInputAndHonorsOverrides() {
    let builtIn = AudioDevice(id: 1, uid: "builtin", name: "Mac microphone", inputs: true, outputs: false)
    let headset = AudioDevice(id: 2, uid: "headset", name: "AirPods", inputs: true, outputs: true)
    let virtual = AudioDevice(id: 3, uid: micUID, name: "CleanMyMic", inputs: true, outputs: false)
    #expect(microphoneSource(["", "headset", micUID], in: [virtual, builtIn, headset]) == headset.id)
    #expect(microphoneSource(["builtin", "headset", micUID], in: [virtual, builtIn, headset]) == builtIn.id)
    #expect(microphoneSource(["headset", "builtin", micUID], in: [virtual, builtIn]) == builtIn.id)
    #expect(microphoneSource([micUID], in: [virtual]) == nil)
}
