import CoreAudio
import Foundation
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

@Test func systemRoutingRequiresOptInAndRespectsManualChoices() {
    let mic = AudioDevice(id: 1, uid: micUID, name: "CleanMyMic", inputs: true, outputs: false)
    let cable = AudioDevice(id: 2, uid: cableUID, name: "Cable", inputs: false, outputs: true)
    let physical = AudioDevice(id: 3, uid: "usb", name: "USB mic", inputs: true, outputs: false)
    let alternative = AudioDevice(id: 4, uid: "other", name: "Other input", inputs: true, outputs: false, isVirtual: true)
    let devices = [mic, cable, physical, alternative]
    func change(_ current: AudioDevice?, _ saved: String?, enabled: Bool, claim: Bool = false) -> SystemInputChange {
        .decide(current: current, devices: devices, previousUID: saved, enabled: enabled, allowClaim: claim)
    }
    #expect(change(physical, nil, enabled: false, claim: true) == .none)
    #expect(change(physical, nil, enabled: true) == .none)
    #expect(change(physical, nil, enabled: true, claim: true) == .claim(mic.id, previousUID: "usb"))
    #expect(change(mic, "usb", enabled: true) == .none)
    #expect(change(mic, "usb", enabled: false) == .restore(3))
    #expect(change(mic, "other", enabled: false) == .restore(4))
    #expect(change(mic, "disconnected", enabled: false) == .restore(3))
    #expect(change(mic, nil, enabled: false) == .none)
    #expect(change(alternative, "usb", enabled: true, claim: true) == .releaseOwnership)
    #expect(change(nil, "usb", enabled: false) == .none)
}

@Test func theAppsOwnRecordingDoesNotCountAsSomeoneUsingTheMic() {
    func alone(_ readers: [(pid: pid_t, uids: [String])]) -> Bool { onlySelfRecords(from: micUID, readers: readers, me: 7) }
    #expect(alone([(7, [micUID])]))
    #expect(!alone([(7, [micUID]), (8, [micUID])]))  // Sound settings showing the level, say
    #expect(!alone([(7, [micUID]), (8, ["CADefaultDeviceAggregate-812-0"])]))  // may wrap the mic: assume it does
    #expect(!alone([(8, [micUID])]))
    #expect(!alone([(7, ["airpods"]), (8, ["usb"])]))  // the normal case: we read the physical mic
    #expect(alone([(7, [micUID]), (8, ["usb"])]))
}

@Test func aHungAudioPathIsNoticedWhetherStartingOrRunning() {
    let now = Date()
    func stuck(_ starting: TimeInterval?, running: Bool = false, healthy: TimeInterval = 0) -> Bool {
        audioIsStuck(startingSince: starting.map { now.addingTimeInterval(-$0) }, running: running,
                     lastHealthy: now.addingTimeInterval(-healthy), now: now)
    }
    #expect(!stuck(5) && stuck(16))  // a slow first model load is fine, a start that never ends is not
    #expect(!stuck(nil, running: true, healthy: 6) && stuck(nil, running: true, healthy: 16))
    #expect(!stuck(nil, running: false, healthy: 600))  // idle and ready is not stuck
}

@Test func stalledPlaybackRetriesOnceUntilSamplesFlowAgain() {
    var watch = PlaybackWatch(recovering: false)
    let firstInterval = watch.advanced(to: 0)
    #expect(!firstInterval && watch.canRestart)
    watch = PlaybackWatch(recovering: true)
    let afterRestart = watch.advanced(to: 0)
    #expect(!afterRestart && !watch.canRestart)
    let delivered = watch.advanced(to: 480)  // queued silence counts as delivered audio
    #expect(delivered && watch.canRestart)
    let stalled = !watch.advanced(to: 480)
    #expect(stalled)
}

@MainActor @Test func delayedSystemChangesRespectOffManualOverrideAndRelaunch() {
    let suite = "dev.vaze.CleanMyMic.tests." + UUID().uuidString
    let prefs = UserDefaults(suiteName: suite)!
    defer { prefs.removePersistentDomain(forName: suite) }
    let mic = AudioDevice(id: 1, uid: micUID, name: "CleanMyMic", inputs: true, outputs: false)
    let cable = AudioDevice(id: 2, uid: cableUID, name: "Cable", inputs: false, outputs: true)
    let usb = AudioDevice(id: 3, uid: "usb", name: "USB microphone", inputs: true, outputs: false)
    let headset = AudioDevice(id: 4, uid: "headset", name: "Headset", inputs: true, outputs: false)
    var current = usb.id
    var writes: [AudioDeviceID] = []
    func launch() -> SystemInputRouting {
        SystemInputRouting(defaults: prefs, devices: { [mic, cable, usb, headset] },
                           current: { current }, select: { writes.append($0) })
    }
    var router = launch()
    router.update(cleanupOn: true, allowClaim: true)
    #expect(writes.isEmpty)
    router.setEnabled(true, cleanupOn: true)
    router.update(cleanupOn: false)  // off before Core Audio confirms the claim
    #expect(writes == [mic.id])
    current = mic.id
    router.update(cleanupOn: false)
    #expect(writes == [mic.id, usb.id])
    // The restore completes after a normal quit. The opt-in survives the next launch.
    current = usb.id
    router = launch()
    router.update(cleanupOn: true, allowClaim: true)
    #expect(router.enabled && writes == [mic.id, usb.id, mic.id])
    current = mic.id
    router.update(cleanupOn: true)
    current = headset.id
    router.update(cleanupOn: true)
    #expect(!router.enabled)
    #expect(prefs.string(forKey: "restoreInput") == nil)
    router.update(cleanupOn: false, release: true)
    #expect(writes == [mic.id, usb.id, mic.id])
}
