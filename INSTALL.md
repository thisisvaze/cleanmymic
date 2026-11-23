# Install CleanMyMic

CleanMyMic requires an Apple Silicon Mac (M1 or newer) and macOS 15 or later. Intel Macs are not supported.

## Build from source

Install Xcode (Swift 6.2 or newer), open Xcode once to finish setup, then compile the app:

```sh
git clone https://github.com/thisisvaze/cleanmymic.git
cd cleanmymic
swift build -c release --package-path mac
```

This verifies the Swift app compiles without installing or launching it.

## Run a local build

Bundle the app with an exported Core ML model package placed at `models/CleanMyMicStudio.mlpackage` (or pass `MODEL=/path/to/model.mlpackage`).

```sh
./mac/bundle.sh
./mac/driver/build.sh
./mac/driver/install.sh
open mac/build/CleanMyMic.app
```

- `./mac/bundle.sh install` copies the signed app straight into `/Applications/CleanMyMic.app` and launches it.
- `./mac/driver/build.sh` fetches pinned [BlackHole](https://github.com/ExistentialAudio/BlackHole) source and compiles `CleanMyMic.driver`.
- `./mac/driver/install.sh` asks for an administrator password and briefly restarts Core Audio - finish active calls and recordings first.
- If you have an Apple Developer ID or Apple Development certificate locally, `bundle.sh` signs the app with it so microphone permissions carry over across rebuilds. Otherwise it signs ad-hoc.

## First use

1. Open **CleanMyMic** from Applications (it lives in the menu bar).
2. In Zoom, Meet, Teams, OBS or another app, select **CleanMyMic** as the microphone. Turn that app's built-in noise suppression off or to Low so it doesn't double-process your voice.
3. Allow microphone access when macOS prompts, then try **Mic check** in the CleanMyMic menu with headphones on to compare **Original** and **Cleaned**.
4. **Ready** means no app is reading from CleanMyMic (your physical mic is released). **In use** means audio is flowing.

The switch at the top right turns cleanup on or off - turning it off passes original audio through without disconnecting the mic. System-wide default input routing is off by default and can be toggled in **Settings > Cleanup**.

## Build a `.pkg` installer

With `models/CleanMyMicStudio.mlpackage` in place:

```sh
# Unsigned local test package
./mac/installer/build.sh --unsigned

# Signed and notarized distribution package
APPLICATION_IDENTITY='Developer ID Application: Your Name (TEAM_ID)' \
INSTALLER_IDENTITY='Developer ID Installer: Your Name (TEAM_ID)' \
NOTARY_PROFILE=CleanMyMic ./mac/installer/build.sh
```

Outputs go to `mac/build/release/`.

## Remove CleanMyMic

Quit CleanMyMic, then remove the virtual audio driver:

```sh
./mac/driver/install.sh remove
```

(If installed from a `.pkg`, run `sudo "/Library/Application Support/CleanMyMic/uninstall.sh"` and restart.) Delete `CleanMyMic.app` from `/Applications`, remove any remaining Login Item in System Settings, and switch your call apps back to a physical microphone.
