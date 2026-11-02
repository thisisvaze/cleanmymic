import SwiftUI

struct MenuView: View {
    @Bindable var controller: Controller
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismiss) private var dismiss

    private var stateLabel: String {
        if controller.isRunning { return controller.status }
        return ["Starting…", "Ready"].contains(controller.status) ? controller.status : "Needs attention"
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "waveform")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(controller.isRunning ? Color.accentColor : .secondary)
                    .frame(width: 32, height: 32)
                    .background(.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 9))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text("CleanMyMic").font(.system(size: 14, weight: .semibold))
                    Text(controller.noiseRemoval || controller.testing ? stateLabel : "Off · \(stateLabel)")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("CleanMyMic", isOn: $controller.noiseRemoval)  // the main switch: what calls get
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .disabled(controller.testing)  // the test has its own Original | Cleaned
                    .help(controller.testing ? "Calls keep this setting during the test"
                          : "Off sends your original voice, uncleaned")
            }
            .padding(16)

            if controller.noiseRemoval {  // off: just the switch, like Bluetooth
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Hear yourself").font(.system(size: 12, weight: .medium))
                        Text(controller.testing ? "Playing in \(controller.headphoneName). Calls aren't affected."
                             : controller.testBlocker ?? "Test in your headphones before a call")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    Toggle("Hear yourself", isOn: $controller.testing)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .controlSize(.small)
                        .disabled(!controller.testing && controller.testBlocker != nil)
                        .help("Plays your microphone to your headphones while this panel is open")
                }
                .padding(.horizontal, 16)
                .padding(.bottom, controller.testing ? 10 : 16)

                if controller.testing {
                    Picker("Compare", selection: $controller.testCleaned) {
                        Text("Original").tag(false)
                        Text("Cleaned").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .help("Only for this test; calls keep the main switch setting")
                    .padding(.horizontal, 16)
                    .padding(.bottom, 16)
                }

                MicrophoneMenu(selection: $controller.inputUID, devices: controller.microphones,
                               deviceName: controller.microphoneName)
                .padding(.horizontal, 12)
                .background(.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.primary.opacity(0.06)))
                .padding(.horizontal, 16)

                Text("Choose CleanMyMic as the microphone in your call app.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(16)
            }

            if stateLabel == "Needs attention" {
                Text(controller.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }

            Divider().padding(.horizontal, 16)
            VStack(spacing: 0) {  // system-style rows, like the Wi-Fi and Bluetooth menus
                MenuRow("Settings…") {
                    dismiss()
                    openWindow(id: "settings")
                    NSApp.activate()  // a menu bar app's window opens behind others otherwise
                }
                .keyboardShortcut(",")
                MenuRow("Quit CleanMyMic") { controller.quit() }
                    .keyboardShortcut("q")
            }
            .padding(6)
        }
        .frame(width: 320)
        .onAppear { controller.refresh() }
        .onDisappear { controller.testing = false }  // never left playing into headphones unseen
    }

}

/// A full-width menu item that highlights on hover.
private struct MenuRow: View {
    let title: String
    let action: () -> Void
    @State private var hovered = false

    init(_ title: String, action: @escaping () -> Void) { (self.title, self.action) = (title, action) }

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 13))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(hovered ? Color.primary.opacity(0.1) : .clear, in: RoundedRectangle(cornerRadius: 5))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
    }
}

private struct MicrophoneMenu: View {
    @Binding var selection: String
    let devices: [AudioDevice]
    let deviceName: String

    var body: some View {
        Menu {
            Picker("Microphone", selection: $selection) {
                Text("Automatic").tag("")
                if !selection.isEmpty && !devices.contains(where: { $0.uid == selection }) {
                    Text("Preferred microphone unavailable").tag(selection)
                }
                ForEach(devices) { Text($0.name).tag($0.uid) }
            }
            .pickerStyle(.inline)
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "mic")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Using").font(.system(size: 10)).foregroundStyle(.secondary)
                    Text(deviceName)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 8)
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .padding(.vertical, 10)
            .contentShape(Rectangle())
            .accessibilityElement(children: .combine)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .accessibilityLabel("Using \(deviceName)")
        .help(deviceName)
    }
}
