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
    private var lastCleanupOn = false
    private var lastPreferredInputUID: String?
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

    func setEnabled(_ value: Bool, cleanupOn: Bool, preferredInputUID: String? = nil) {
        enabled = value
        defaults.set(value, forKey: "useSystemInput")
        error = nil
        update(cleanupOn: cleanupOn, preferredInputUID: preferredInputUID, allowClaim: value)
    }

    /// Claims the Mac's default input when opted in, and always switches away from CleanMyMic when off or quitting.
    func update(cleanupOn: Bool, preferredInputUID: String? = nil, allowClaim: Bool = false, release: Bool = false) {
        lastCleanupOn = cleanupOn && !release
        if let preferredInputUID { lastPreferredInputUID = preferredInputUID }
        wantsSystemInput = cleanupOn && enabled && !release
        let leaveVirtualInput = !cleanupOn || release
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
                    return
                }
                if !(release && current.uid == micUID) {
                    return
                }
            }
            self.pendingInput = nil
            routingCheck?.cancel()
        }
        let change = SystemInputChange.decide(current: current, devices: all,
                                              previousUID: defaults.string(forKey: "restoreInput"),
                                              preferredInputUID: lastPreferredInputUID,
                                              enabled: wantsSystemInput, allowClaim: requestedClaim,
                                              leaveVirtualInput: leaveVirtualInput)
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
                        update(cleanupOn: lastCleanupOn, preferredInputUID: lastPreferredInputUID)
                    }
                }
            }
        } catch {
            requestedClaim = false
            self.error = String(describing: error)
        }
    }
}
