import CoreAudio
import Observation
import Foundation

@Observable
final class SystemInputRouting {
    private let defaults: UserDefaults
    private let availableDevices: () -> [AudioDevice]
    private let currentInput: () -> AudioDeviceID?
    private let selectInput: (AudioDeviceID) throws -> Void
    private var wantsSystemInput = false
    private(set) var enabled: Bool
    private(set) var error: String?
    private var requestedClaim = false
    private var pendingInput: (from: AudioDeviceID, to: AudioDeviceID)?
    private var routingCheck: Task<Void, Never>?

    init(defaults: UserDefaults = .standard, devices: @escaping () -> [AudioDevice] = audioDevices,
         current: @escaping () -> AudioDeviceID? = { defaultDevice(input: true) },
         select: @escaping (AudioDeviceID) throws -> Void = setDefaultInput) {
        self.defaults = defaults
        availableDevices = devices
        currentInput = current
        selectInput = select
        enabled = defaults.bool(forKey: "useSystemInput")
    }

    func setEnabled(_ value: Bool, cleanupOn: Bool) {
        enabled = value
        defaults.set(value, forKey: "useSystemInput")
        error = nil
        update(cleanupOn: cleanupOn, allowClaim: value)
    }

    /// Only the app that took the default may restore it. Older builds used the same ownership key.
    func update(cleanupOn: Bool, allowClaim: Bool = false, release: Bool = false) {
        wantsSystemInput = cleanupOn && enabled && !release
        let all = availableDevices()
        if allowClaim { requestedClaim = true }
        if !wantsSystemInput { requestedClaim = false }
        guard let id = currentInput(), let current = all.first(where: { $0.id == id }) else { return }
        // A normal quit can finish restoring the input after the app exits.
        if defaults.string(forKey: "restoringTo") == current.uid {
            defaults.removeObject(forKey: "restoreInput")
            defaults.removeObject(forKey: "restoringTo")
        }
        if let pendingInput {
            // Core Audio can acknowledge a write before it changes the default.
            if id == pendingInput.from {
                if release && all.contains(where: { $0.id == pendingInput.to && $0.uid == micUID }) {
                    // A quit/failure can arrive before our claim. Queue its inverse before leaving.
                    do {
                        try selectInput(id)
                        defaults.set(current.uid, forKey: "restoringTo")
                    } catch { self.error = String(describing: error) }
                }
                return
            }
            self.pendingInput = nil
            routingCheck?.cancel()
        }
        let change = SystemInputChange.decide(current: current, devices: all,
                                              previousUID: defaults.string(forKey: "restoreInput"),
                                              enabled: wantsSystemInput, allowClaim: requestedClaim)
        do {
            switch change {
            case .none:
                if current.uid == micUID { requestedClaim = false }
                return
            case let .claim(target, previous):
                try selectInput(target)
                defaults.removeObject(forKey: "restoringTo")
                defaults.set(previous, forKey: "restoreInput")
                pendingInput = (id, target)
            case let .restore(target):
                try selectInput(target)
                defaults.set(all.first { $0.id == target }?.uid, forKey: "restoringTo")
                pendingInput = (id, target)
            case .releaseOwnership:
                defaults.removeObject(forKey: "restoringTo")
                defaults.removeObject(forKey: "restoreInput")
                enabled = false
                defaults.set(false, forKey: "useSystemInput")
            }
            requestedClaim = false
            self.error = nil
            if pendingInput != nil {
                routingCheck?.cancel()
                routingCheck = Task { [weak self] in
                    do { try await Task.sleep(for: .seconds(1)) } catch { return }
                    guard let self else { return }
                    if let request = pendingInput, currentInput() == request.from {
                        pendingInput = nil
                        self.error = "The Mac’s input did not switch. Check Sound settings."
                    } else {
                        update(cleanupOn: wantsSystemInput)
                    }
                }
            }
        } catch {
            requestedClaim = false
            self.error = String(describing: error)
        }
    }
}
