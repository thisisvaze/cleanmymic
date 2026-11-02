import SwiftUI

/// The menu bar panel, laid out like the system's Wi-Fi and Bluetooth menus with standard controls only.
struct MenuView: View {
    @Bindable var controller: Controller
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismiss) private var dismiss

    private var stateLabel: String {
        if controller.isRunning { return controller.status }
        return ["Starting…", "Ready"].contains(controller.status) ? controller.status : "Needs attention"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("CleanMyMic").font(.headline)
                    Text(controller.noiseRemoval ? stateLabel : "Off · \(stateLabel)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("CleanMyMic", isOn: $controller.noiseRemoval)  // the main switch: what calls get
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .help("Off sends your original voice, uncleaned, and ends Hear yourself")
            }
            .padding(12)

            if stateLabel == "Needs attention" {
                Text(controller.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding([.horizontal, .bottom], 12)
            }

            if controller.noiseRemoval {  // off: just the switch, like Bluetooth
                Divider().padding(.horizontal, 12)
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Microphone", selection: $controller.inputUID) {
                        Text("Automatic").tag("")
                        if !controller.inputUID.isEmpty
                            && !controller.microphones.contains(where: { $0.uid == controller.inputUID }) {
                            Text("Preferred microphone unavailable").tag(controller.inputUID)
                        }
                        ForEach(controller.microphones) { Text($0.name).tag($0.uid) }
                    }
                    .help("Using \(controller.microphoneName)")
                    Text("Choose CleanMyMic as the microphone in your call app.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(12)

                Divider().padding(.horizontal, 12)
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Hear yourself")
                        Spacer()
                        Toggle("Hear yourself", isOn: $controller.testing)
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .disabled(!controller.testing && controller.testBlocker != nil)
                    }
                    Text(controller.testing ? "Playing in \(controller.headphoneName). Calls aren't affected."
                         : controller.testBlocker ?? "Test in your headphones before a call.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if controller.testing {
                        Picker("Compare", selection: $controller.testCleaned) {
                            Text("Original").tag(false)
                            Text("Cleaned").tag(true)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                    }
                }
                .padding(12)
            }

            Divider().padding(.horizontal, 12)
            VStack(spacing: 0) {
                MenuRow("Settings…") {
                    dismiss()
                    openWindow(id: "settings")
                    NSApp.activate()  // a menu bar app's window opens behind others otherwise
                }
                .keyboardShortcut(",")
                MenuRow("Quit CleanMyMic") { controller.quit() }
                    .keyboardShortcut("q")
            }
            .padding(5)
        }
        .frame(width: 300)
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
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(.fill.tertiary.opacity(hovered ? 1 : 0), in: .rect(cornerRadius: 5))
                .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
    }
}
