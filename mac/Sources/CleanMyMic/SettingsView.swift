import SwiftUI

/// The Settings window: a sidebar of panes, like System Settings.
struct SettingsView: View {
    @Bindable var controller: Controller
    @State private var pane: Pane? = .cleanup

    private enum Pane: String, CaseIterable, Identifiable {
        case cleanup = "Cleanup", model = "Model", about = "About"
        var id: Self { self }
        var icon: String {
            switch self {
            case .cleanup: "slider.horizontal.3"
            case .model: "cpu"
            case .about: "info.circle"
            }
        }
    }

    var body: some View {
        NavigationSplitView {
            List(Pane.allCases, selection: $pane) { Label($0.rawValue, systemImage: $0.icon) }
                .navigationSplitViewColumnWidth(170)
                .toolbar(removing: .sidebarToggle)
        } detail: {
            Form {
                switch pane ?? .cleanup {
                case .cleanup: cleanup
                case .model: model
                case .about: about
                }
            }
            .formStyle(.grouped)
            .navigationTitle((pane ?? .cleanup).rawValue)
        }
        .frame(width: 600, height: 360)
    }

    @ViewBuilder private var cleanup: some View {
        Section {
            LabeledContent("Strength") {
                HStack {
                    Slider(value: $controller.strength, in: 0...1)
                        .labelsHidden()
                        .accessibilityLabel("Cleanup strength")
                    Text(controller.strength, format: .percent.precision(.fractionLength(0)))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .frame(width: 40, alignment: .trailing)
                }
            }
        } footer: {
            Text("How much background noise to remove when CleanMyMic is on. Lower keeps more of your voice's character.")
        }
    }

    @ViewBuilder private var model: some View {
        Section {
            Picker("Model", selection: $controller.model) {
                ForEach(controller.models, id: \.self) { Text(modelDisplayName($0)).tag($0) }
            }
            .pickerStyle(.inline)
            .labelsHidden()
        } footer: {
            Text("Runs on this Mac with Core ML. Switching restarts the microphone for a moment.")
        }
    }

    @ViewBuilder private var about: some View {
        Section {
            LabeledContent("Version",
                           value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "–")
            LabeledContent("Processing", value: "On-device, no account or cloud")
            Link("CleanMyMic on GitHub", destination: URL(string: "https://github.com/thisisvaze/cleanmymic")!)
        }
    }
}
