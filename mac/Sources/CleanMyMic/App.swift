// CleanMyMic stays ready in the menu bar, capturing only while another app opens its virtual microphone.
import AVFoundation
import CoreAudio
import ServiceManagement
import SwiftUI

@main
struct CleanMyMicApp: App {
    @State private var controller = Controller()

    init() {
        let args = CommandLine.arguments
        if args.count == 4, args[1] == "--check" { check(mic: args[2], output: args[3]) }
    }

    var body: some Scene {
        MenuBarExtra {
            MenuView(controller: controller)
        } label: {
            Image(systemName: controller.menuBarSymbol)
                .accessibilityLabel("CleanMyMic")
        }
        .menuBarExtraStyle(.window)

        Window("CleanMyMic Settings", id: "settings") {
            SettingsView(controller: controller)
        }
        .windowResizability(.contentSize)
        .defaultLaunchBehavior(.suppressed)  // opened from the menu, never at launch
    }
}

private nonisolated let defaultModel = "Hold-10k"

/// The models the app bundles, by resource name. Only Base ships today; others appear here when bundled.
nonisolated func bundledModels(_ bundle: Bundle = .main) -> [String] {
    (bundle.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? [])
        .map { $0.deletingPathExtension().lastPathComponent }.sorted()
}

nonisolated func modelDisplayName(_ resource: String) -> String {
    resource == defaultModel ? "CleanMyMic Base" : resource
}

@Observable
final class Controller {
    private let defaults = UserDefaults.standard
    var noiseRemoval: Bool {
        didSet {
            defaults.set(noiseRemoval, forKey: "enabled")
            if !noiseRemoval { testing = false }  // off ends the headphone test (its didSet restarts)
            pipeline?.setMix(mix)
            if oldValue != noiseRemoval {
                systemRouting.update(cleanupOn: noiseRemoval, preferredInputUID: inputUID, allowClaim: noiseRemoval)
                if noiseRemoval && needsAttention { restart() }
            }
        }
    }
    let systemRouting = SystemInputRouting()
    func setUseSystemInput(_ enabled: Bool) {
        systemRouting.setEnabled(enabled, cleanupOn: noiseRemoval, preferredInputUID: inputUID)
    }
    var inputUID: String {  // "": automatic
        didSet {
            guard inputUID != oldValue else { return }
            defaults.set(inputUID, forKey: "input")
            defaults.set(devices.first { $0.uid == inputUID }?.name, forKey: "inputName")  // to name it while it's away
            restart()
        }
    }
    /// The chosen mic is disconnected (AirPods in their case): another one stands in until it's back.
    var preferredMissing: Bool { !inputUID.isEmpty && !microphones.contains { $0.uid == inputUID } }
    var preferredName: String { defaults.string(forKey: "inputName") ?? "Your chosen microphone" }
    var strength: Double { didSet { defaults.set(strength, forKey: "strength"); pipeline?.setMix(mix) } }
    var model: String { didSet { if model != oldValue { defaults.set(model, forKey: "model"); restart() } } }
    let models = bundledModels()
    /// Mic check: hear yourself through headphones, to compare original and cleaned. A call app opening the mic ends it.
    var testing = false { didSet { if testing != oldValue { testCleaned = true; restart() } } }
    /// The test's own comparison: never changes `noiseRemoval`, which is what calls get.
    var testCleaned = true { didSet { pipeline?.setMix(mix) } }
    var devices = audioDevices()
    private(set) var headphones = defaultDevice(input: false)  // where a test plays: the system output
    private(set) var status = "Starting…" {
        didSet { startingSince = status == "Starting…" ? startingSince ?? Date() : nil }
    }
    private var startingSince: Date?  // how long the app has been trying to start, for the stuck check
    private var lastHealthy = Date()  // the last pass of the running pipeline's health loop
    private(set) var isRunning = false
    private var pipeline: Pipeline?
    private var cleaning: AudioDeviceID?  // the mic in use, or being started
    private var destination: AudioDeviceID?
    private var observedMic: AudioDeviceID?
    private var activityListener: AudioObjectPropertyListenerBlock?
    private var consumerActive = false
    private var generation = 0  // a newer restart() makes an older, still-awaiting start() give up
    private var startTask: Task<Void, Never>?

    init() {
        noiseRemoval = defaults.object(forKey: "enabled") as? Bool ?? true
        inputUID = defaults.string(forKey: "input") ?? ""
        strength = defaults.object(forKey: "strength") as? Double ?? 1
        let saved = defaults.string(forKey: "model") ?? defaultModel
        model = models.contains(saved) ? saved : defaultModel  // a model dropped from the app falls back to Base
        try? SMAppService.mainApp.register()  // start at login
        // Ask for the mic up front: the system prompt takes focus, which would close the menu panel mid-toggle.
        if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
            Task { _ = await AVCaptureDevice.requestAccess(for: .audio) }
        }
        watch(kAudioHardwarePropertyDefaultInputDevice) { [weak self] in Task { @MainActor in self?.devicesChanged() } }
        watch(kAudioHardwarePropertyDevices) { [weak self] in Task { @MainActor in self?.devicesChanged() } }
        watch(kAudioHardwarePropertyDefaultOutputDevice) { [weak self] in Task { @MainActor in self?.outputChanged() } }
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.systemRouting.update(cleanupOn: false, preferredInputUID: self?.inputUID, release: true)
            }
        }
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification,
                                                          object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.restart() }
        }
        systemRouting.update(cleanupOn: noiseRemoval, preferredInputUID: inputUID, allowClaim: true)
        Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.checkStuck() }
        }
        restart()
    }

    /// A hung audio path cannot be stopped from inside, and reopening the app has cleared it every time: do that.
    private func checkStuck() {
        guard audioIsStuck(startingSince: startingSince, running: isRunning, lastHealthy: lastHealthy, now: Date()) else { return }
        let last = defaults.object(forKey: "relaunched") as? Date ?? .distantPast
        guard Date().timeIntervalSince(last) > 60 else {  // it did not help a minute ago: say so and stop trying
            isRunning = false
            status = "Audio is stuck. Quit and reopen CleanMyMic."
            return
        }
        defaults.set(Date(), forKey: "relaunched")
        let fresh = NSWorkspace.OpenConfiguration()
        fresh.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: fresh) { _, _ in exit(0) }
    }

    var needsAttention: Bool { !isRunning && !["Starting…", "Ready"].contains(status) }
    var isSystemInput: Bool { devices.first { $0.id == defaultDevice(input: true) }?.uid == micUID }
    var menuBarSymbol: String {
        if !noiseRemoval { return "waveform.slash" }
        if needsAttention { return "exclamationmark.triangle" }
        return isRunning ? "waveform.circle.fill" : "waveform"
    }

    var headphoneName: String { devices.first { $0.id == headphones }?.name ?? "your headphones" }
    var testBlocker: String? {
        headphoneTestBlocker(output: devices.first { $0.id == headphones },
                             isSpeaker: headphones.map(isBuiltInSpeaker) ?? false, inCall: isRunning && !testing)
    }

    private func outputChanged() {
        devices = audioDevices()
        headphones = defaultDevice(input: false)
        guard testing else { return }
        if testBlocker == nil { restart() } else { testing = false }  // e.g. headphones unplugged: off, not speakers
    }

    var microphones: [AudioDevice] { devices.filter(\.isMicrophoneSource) }
    var microphoneName: String { devices.first { $0.id == source() }?.name ?? "No microphone found" }
    private var mix: Float { (testing ? testCleaned : noiseRemoval) ? Float(strength) : 0 }

    private func source() -> AudioDeviceID? {
        let system = devices.first { $0.id == defaultDevice(input: true) }?.uid
        return microphoneSource([inputUID, system, defaults.string(forKey: "restoreInput")], in: devices)
    }

    /// A device came or went: the mic in use, or the one we've been waiting for.
    private func devicesChanged() {
        devices = audioDevices()
        headphones = defaultDevice(input: false)
        systemRouting.update(cleanupOn: noiseRemoval, preferredInputUID: inputUID)
        if source() != cleaning || cleanMicOutput(in: devices)?.id != destination
            || devices.first(where: { $0.uid == micUID })?.id != observedMic { restart() }
    }

    /// The public input and hidden output are separate devices: our writer must not count as a consumer.
    private func observeMic(_ device: AudioDeviceID?) throws {
        guard observedMic != device else { return }
        var property = address(kAudioDevicePropertyDeviceIsRunningSomewhere)
        if let old = observedMic, let listener = activityListener {
            AudioObjectRemovePropertyListenerBlock(old, &property, .main, listener)
        }
        observedMic = nil
        activityListener = nil
        consumerActive = false
        guard let device else { return }
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            Task { @MainActor in self?.micUseChanged() }
        }
        let result = AudioObjectAddPropertyListenerBlock(device, &property, .main, listener)
        guard result == noErr else { throw AudioError.activity(result) }
        observedMic = device
        activityListener = listener
    }

    private func micUseChanged() {
        guard let device = observedMic else { return }
        do {
            let active = try otherAppIsRecording(from: device)
            guard active != consumerActive else { return }
            consumerActive = active
            restart()
        } catch {
            failed(String(describing: error), generation)
        }
    }

    func quit() {
        try? observeMic(nil)
        NSApplication.shared.terminate(nil)
    }

    func refresh() { devicesChanged() }

    private func restart(recovering: Bool = false) {
        generation += 1
        let previous = startTask
        previous?.cancel()
        let gen = generation, old = pipeline
        pipeline = nil
        isRunning = false
        status = "Starting…"
        startTask = Task {
            await previous?.value  // finish stopping any superseded engine before opening the next one
            await Self.stop(old)
            guard gen == generation else { return }
            await start(gen, recovering: recovering)
        }
    }

    private func start(_ gen: Int, recovering: Bool) async {
        do {
            devices = audioDevices()
            guard let output = cleanMicOutput(in: devices), let mic = devices.first(where: { $0.uid == micUID }) else {
                try observeMic(nil)
                failed("Install the CleanMyMic virtual microphone to use it in calls.", gen)
                return
            }
            guard let input = source() else { failed("No microphone found.", gen); return }
            guard let model = Bundle.main.url(forResource: model, withExtension: "mlmodelc") else {
                failed(AudioError.noModel(modelDisplayName(self.model)).description, gen)
                return
            }
            try observeMic(mic.id)
            cleaning = input
            destination = output.id
            consumerActive = try otherAppIsRecording(from: mic.id)
            if testing && (consumerActive || testBlocker != nil) { testing = false; return }  // didSet restarts
            guard consumerActive || testing else { status = "Ready"; return }
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            guard gen == generation else { return }
            guard granted else {
                failed("Allow the mic: System Settings > Privacy & Security > Microphone", gen)
                return
            }
            consumerActive = try otherAppIsRecording(from: mic.id)
            if testing && consumerActive { testing = false; return }
            guard consumerActive || testing, let target = testing ? headphones : output.id else {
                status = "Ready"
                return
            }
            let onError: @Sendable (String) -> Void = { [weak self] message in
                Task { @MainActor in self?.recover(message, gen) }
            }
            playbackWatch = PlaybackWatch(recovering: recovering)
            let p = try await Self.makePipeline(model: model, input: input, output: target, mix: mix,
                                                onError: onError)
            guard gen == generation else { return await Self.stop(p) }  // switched again while this one loaded
            pipeline = p
            p.setMix(mix)
            lastHealthy = Date()
            isRunning = true
            status = testing ? "Testing" : "In use"
            while !Task.isCancelled {
                try await Task.sleep(for: .seconds(3))
                if !testing, let observedMic, try !otherAppIsRecording(from: observedMic) {
                    restart()
                    return
                }
                guard await Self.onDevices(p) else {
                    recover("CleanMyMic lost its microphone or output.", gen)
                    return
                }
                lastHealthy = Date()
                if !playbackWatch.advanced(to: p.samplesDelivered) {
                    recover("Audio stopped reaching the selected output.", gen)
                    return
                }
            }
        } catch {
            failed(String(describing: error), gen)
        }
    }

    /// Loading a Core ML model (the Neural Engine compiles it on first use) and starting or stopping the audio
    /// engines takes up to seconds: never on the main actor, or the menu freezes.
    @concurrent
    private nonisolated static func makePipeline(model: URL, input: AudioDeviceID, output: AudioDeviceID, mix: Float,
                                                 onError: @escaping @Sendable (String) -> Void) async throws -> Pipeline {
        let p = try Pipeline(model: model, input: input, output: output, mix: mix, onError: onError)
        try p.start()
        return p
    }

    @concurrent
    private nonisolated static func onDevices(_ p: Pipeline) async -> Bool { p.onIntendedDevices }

    private var playbackWatch = PlaybackWatch(recovering: false)

    private func recover(_ message: String, _ gen: Int) {
        guard gen == generation else { return }
        if playbackWatch.canRestart { restart(recovering: true) }
        else { failed(message, gen) }
    }

    /// The running pipeline gave up on a device change.
    private func failed(_ message: String, _ gen: Int) {
        guard gen == generation else { return }
        generation += 1
        let previous = startTask
        previous?.cancel()
        let old = pipeline
        pipeline = nil
        status = message
        isRunning = false
        cleaning = nil  // the next device change tries again
        destination = nil
        systemRouting.update(cleanupOn: false, preferredInputUID: inputUID, release: true)
        startTask = Task {
            await previous?.value
            await Self.stop(old)
        }
    }

    @concurrent
    private nonisolated static func stop(_ p: Pipeline?) async { p?.stop() }
}

/// Internal routing test: --check "<mic>" "<output>" switches cleanup off/on in each of two four-second runs.
func check(mic: String, output: String) -> Never {
    let devices = audioDevices()
    guard let input = devices.first(where: { $0.inputs && $0.name == mic }),
          let out = devices.first(where: { $0.outputs && $0.name == output }) else {
        print("devices:", devices.map { "\($0.name) (\($0.inputs ? "in" : "")\($0.outputs ? " out" : ""))" })
        exit(1)
    }
    let modelName = modelDisplayName(defaultModel)
    guard let model = Bundle.main.url(forResource: defaultModel, withExtension: "mlmodelc") else {
        print(AudioError.noModel(modelName))
        exit(1)
    }
    for _ in 0..<2 {
        do {
            let p = try Pipeline(model: model, input: input.id, output: out.id, mix: 1) { print("pipeline error:", $0) }
            try p.start()
            var perSecond = [Int](), done = 0
            for mix: Float in [1, 0, 1, 0] {
                p.setMix(mix)
                Thread.sleep(forTimeInterval: 1)
                perSecond.append(p.framesDone - done)
                done = p.framesDone
            }
            precondition(perSecond.allSatisfy { $0 > 0 }, "Audio stopped while switching noise removal")
            print(modelName, p.framesDone, "frames in 4 s (on/off/on/off)", perSecond,
                  String(format: "%.2f ms per frame", p.msPerFrame))
            p.stop()
        } catch {
            print(modelName, "failed:", error)
            exit(1)
        }
    }
    exit(0)
}
