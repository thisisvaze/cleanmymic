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
            Image(systemName: "waveform")
                .accessibilityLabel("CleanMyMic")
        }
        .menuBarExtraStyle(.window)
    }
}

private let modelName = "CleanMyMic Base"
private let modelResource = "Hold-10k"

@Observable
final class Controller {
    private let defaults = UserDefaults.standard
    var noiseRemoval: Bool { didSet { defaults.set(noiseRemoval, forKey: "enabled"); pipeline?.setMix(mix) } }
    var inputUID: String { didSet { defaults.set(inputUID, forKey: "input"); restart() } }  // "": automatic
    var strength: Double { didSet { defaults.set(strength, forKey: "strength"); pipeline?.setMix(mix) } }
    var devices = audioDevices()
    private(set) var status = "Starting…"
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
        try? SMAppService.mainApp.register()  // start at login
        watch(kAudioHardwarePropertyDefaultInputDevice) { [weak self] in Task { @MainActor in self?.systemMicChanged() } }
        watch(kAudioHardwarePropertyDevices) { [weak self] in Task { @MainActor in self?.devicesChanged() } }
        restart()
    }

    var microphones: [AudioDevice] { devices.filter(\.isMicrophoneSource) }
    var microphoneName: String { devices.first { $0.id == source() }?.name ?? "No microphone found" }
    private var mix: Float { noiseRemoval ? Float(strength) : 0 }

    /// Keep one call-app device for both cleaned and original voice; remember the physical mic across restarts.
    private func route() throws {
        guard let mic = devices.first(where: { $0.uid == micUID && $0.inputs })?.id else { return }
        guard let current = defaultDevice(input: true), current != mic else { return }
        defaults.set(devices.first { $0.id == current }?.uid, forKey: "restoreInput")
        try setDefaultInput(mic)
    }

    private func unroute() {
        let saved = defaults.string(forKey: "restoreInput")
        defaults.removeObject(forKey: "restoreInput")
        let all = audioDevices()
        guard all.contains(where: { $0.uid == micUID && $0.id == defaultDevice(input: true) }) else { return }
        if let real = microphoneSource([saved], in: all) { try? setDefaultInput(real) }
    }

    /// The mic to clean: the one last picked, or, while that's gone (AirPods in their case), the next best.
    private func source() -> AudioDeviceID? {
        let system = devices.first { $0.id == defaultDevice(input: true) }?.uid
        return microphoneSource([inputUID, defaults.string(forKey: "restoreInput"), system], in: devices)
    }

    /// Someone picked a mic in Sound, or macOS did (AirPods connected): that's the one to clean now, and the system's
    /// mic goes back to ours. Our own switch lands here too, and stops at the first guard's mic check.
    private func systemMicChanged() {
        devices = audioDevices()
        guard observedMic != nil, let picked = devices.first(where: { $0.id == defaultDevice(input: true) }),
              picked.isMicrophoneSource else { return }
        defaults.set(picked.uid, forKey: "restoreInput")
        if source() != cleaning { restart() }
        else { do { try route() } catch { failed(String(describing: error), generation) } }
    }

    /// A device came or went: the mic in use, or the one we've been waiting for.
    private func devicesChanged() {
        devices = audioDevices()
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
            let active = try deviceIsRunning(device)
            guard active != consumerActive else { return }
            consumerActive = active
            restart()
        } catch {
            failed(String(describing: error), generation)
        }
    }

    func quit() {
        try? observeMic(nil)
        unroute()
        NSApplication.shared.terminate(nil)
    }

    func refresh() { devicesChanged() }

    private func restart() {
        generation += 1
        startTask?.cancel()
        let gen = generation, old = pipeline
        pipeline = nil
        isRunning = false
        status = "Starting…"
        startTask = Task { await start(gen, after: old) }
    }

    private func start(_ gen: Int, after old: Pipeline?) async {
        await Self.stop(old)
        guard gen == generation else { return }
        do {
            devices = audioDevices()
            guard let output = cleanMicOutput(in: devices), let mic = devices.first(where: { $0.uid == micUID }) else {
                try observeMic(nil)
                failed("Install the CleanMyMic virtual microphone to use it in calls.", gen)
                return
            }
            guard let input = source() else { failed("No microphone found.", gen); return }
            guard let model = Bundle.main.url(forResource: modelResource, withExtension: "mlmodelc") else {
                failed(AudioError.noModel(modelName).description, gen)
                return
            }
            try observeMic(mic.id)
            cleaning = input
            destination = output.id
            try route()
            consumerActive = try deviceIsRunning(mic.id)
            guard consumerActive else { status = "Ready"; return }
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            guard gen == generation else { return }
            guard granted else {
                failed("Allow the mic: System Settings > Privacy & Security > Microphone", gen)
                return
            }
            consumerActive = try deviceIsRunning(mic.id)
            guard consumerActive else { status = "Ready"; return }
            let onError: @Sendable (String) -> Void = { [weak self] message in
                Task { @MainActor in self?.failed(message, gen) }
            }
            let p = try await Self.makePipeline(model: model, input: input, output: output.id, mix: mix,
                                                onError: onError)
            guard gen == generation else { return await Self.stop(p) }  // switched again while this one loaded
            pipeline = p
            p.setMix(mix)
            isRunning = true
            status = "In use"
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

    /// The running pipeline gave up on a device change.
    private func failed(_ message: String, _ gen: Int) {
        guard gen == generation else { return }
        generation += 1
        startTask?.cancel()
        let old = pipeline
        pipeline = nil
        status = message
        isRunning = false
        cleaning = nil  // the next device change tries again
        destination = nil
        try? observeMic(nil)
        unroute()  // don't leave apps on a silent cable
        Task { await Self.stop(old) }
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
    guard let model = Bundle.main.url(forResource: modelResource, withExtension: "mlmodelc") else {
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
