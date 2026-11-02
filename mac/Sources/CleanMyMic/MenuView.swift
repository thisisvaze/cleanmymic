import SwiftUI

struct MenuView: View {
    @Bindable var controller: Controller
    @State private var showsSettings = false

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
                    Text(stateLabel)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(16)

            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Noise removal").font(.system(size: 12, weight: .medium))
                    Text(controller.noiseRemoval ? "Reduce background noise" : "Your original voice passes through")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("Noise removal", isOn: $controller.noiseRemoval)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .help("Off sends your original voice without noise removal")
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)

            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Hear yourself").font(.system(size: 12, weight: .medium))
                    Text(controller.testing ? "Playing in \(controller.headphoneName)"
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
                .help("Only for this test; calls keep the Noise removal setting")
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

            if stateLabel == "Needs attention" {
                Text(controller.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }

            Divider()
            HStack {
                Label("On-device", systemImage: "lock")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Settings", systemImage: "gearshape") { showsSettings.toggle() }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .padding(6)
                    .help("Settings")
                    .popover(isPresented: $showsSettings) { settings }
                Button("Quit") { controller.quit() }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .keyboardShortcut("q")
                    .accessibilityLabel("Quit CleanMyMic")
                    .padding(.vertical, 6)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
        .frame(width: 320)
        .onAppear { controller.refresh() }
        .onDisappear { controller.testing = false }  // never left playing into headphones unseen
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Settings").font(.headline)
            HStack {
                Text("Noise removal strength").font(.system(size: 12))
                Spacer()
                Text(controller.strength, format: .percent.precision(.fractionLength(0)))
                    .font(.system(size: 11).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            }
            Slider(value: $controller.strength, in: 0...1)
                .controlSize(.small)
                .accessibilityLabel("Noise removal strength")
                .accessibilityValue(Text(controller.strength, format: .percent.precision(.fractionLength(0))))
            Text("Used when noise removal is on.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .frame(width: 280)
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
