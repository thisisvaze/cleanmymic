import SwiftUI

/// Compact native menu bar panel (Wi-Fi / Bluetooth popover proportions).
struct MenuView: View {
    @Bindable var controller: Controller
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismiss) private var dismiss

    private var stateLabel: String {
        if controller.isRunning { return controller.status }
        return ["Starting…", "Ready"].contains(controller.status) ? controller.status : "Needs attention"
    }

    private var subtitle: String {
        if controller.noiseRemoval { return stateLabel }
        if controller.isRunning { return "Off · Passthrough" }
        return stateLabel == "Ready" ? "Off" : "Off · \(stateLabel)"
    }

    private var micNote: String? {
        if controller.preferredMissing { return "Using \(controller.microphoneName)" }
        if controller.inputUID.isEmpty { return controller.microphoneName }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("CleanMyMic").font(.callout.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("CleanMyMic", isOn: $controller.noiseRemoval)  // the main switch: what calls get
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .help("Off switches Mac Sound away from CleanMyMic and passes original audio to apps using it")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)

            if let error = controller.systemRouting.error {
                Text(error).font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 10).padding(.bottom, 6)
            }
            if stateLabel == "Needs attention" {
                Text(controller.status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 10).padding(.bottom, 6)
            }

            if controller.noiseRemoval {
                Divider().padding(.horizontal, 10)
                VStack(alignment: .leading, spacing: 4) {
                    Picker("Mic", selection: $controller.inputUID) {
                        Text("Automatic").tag("")
                        if controller.preferredMissing {
                            Text("\(controller.preferredName) (offline)").tag(controller.inputUID)
                        }
                        ForEach(controller.microphones) { Text($0.name).tag($0.uid) }
                    }
                    .controlSize(.small)
                    .help("Using \(controller.microphoneName) · Choose CleanMyMic in your call app")
                    if let micNote {
                        Text(micNote)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)

                Divider().padding(.horizontal, 10)
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Mic check").font(.callout)
                        Spacer()
                        Toggle("Mic check", isOn: $controller.testing)
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .controlSize(.small)
                            .disabled(!controller.testing && controller.testBlocker != nil)
                            .help("Hear your mic in headphones before a call")
                    }
                    if controller.testing {
                        Picker("Compare", selection: $controller.testCleaned) {
                            Text("Original").tag(false)
                            Text("Cleaned").tag(true)
                        }
                        .pickerStyle(.segmented)
                        .controlSize(.small)
                        .labelsHidden()
                        Text("In \(controller.headphoneName)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    } else if let blocker = controller.testBlocker {
                        Text(blocker)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
            }

            Divider().padding(.horizontal, 10)
            VStack(spacing: 0) {
                MenuRow("Settings…") {
                    dismiss()
                    openWindow(id: "settings")
                    NSApp.activate()  // a menu bar app's window opens behind others otherwise
                }
                .keyboardShortcut(",")
                MenuRow("Quit") { controller.quit() }
                    .keyboardShortcut("q")
            }
            .padding(4)
        }
        .frame(width: 236)
        .onAppear { controller.refresh() }
        .onDisappear { controller.testing = false }  // never left playing into headphones unseen
    }
}

/// A full-width menu item with the system's hover highlight.
private struct MenuRow: View {
    let title: String
    let action: () -> Void
    @State private var hovered = false

    init(_ title: String, action: @escaping () -> Void) { (self.title, self.action) = (title, action) }

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.callout)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(.fill.tertiary.opacity(hovered ? 1 : 0), in: .rect(cornerRadius: 5))
                .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
    }
}
