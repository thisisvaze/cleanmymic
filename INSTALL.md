# Install CleanMyMic

**Experimental preview.** The [latest release](https://github.com/thisisvaze/cleanmymic/releases/latest) contains `CleanMyMic-0.1.0-alpha.1-arm64.pkg`: the app, virtual microphone, and Base model in one file, Developer ID signed and notarized by Apple.

CleanMyMic requires an Apple Silicon Mac (M1 or newer) and macOS 15 or later. Check **Apple menu → About This Mac**. Intel Macs are not supported.

## Installer preview

The package installs everything locally; no Python, Xcode, model download, or cloud account is needed on the destination Mac. Do not disable Gatekeeper or strip quarantine to force installation.

To install:

1. Finish calls and recordings, then open the `.pkg`. Follow macOS Installer and enter your administrator password when asked.
2. Restart your Mac so the virtual microphone loads.
3. Open **CleanMyMic** in Applications and follow [First use](#first-use).

The release also includes `SHA256SUMS.txt` and the driver's source archive. Compare the package's `shasum -a 256` output with its entry in that checksum file. A checksum checks the file, not Apple signing or notarization. [Signing status and build recipe](docs/releasing.md#build-the-installer).

The installer replaces CleanMyMic's app and driver at their standard paths and preserves preferences. Clean-Mac installation, upgrade, and removal still need testing.

## Build the source

Install Xcode with Swift 6.2 or newer, open Xcode once to complete its setup, then run:

```sh
git clone https://github.com/thisisvaze/cleanmymic.git
cd cleanmymic
swift build -c release --package-path mac
```

This compiles the app without installing or launching it. Python is only needed for [export and evaluation](CONTRIBUTING.md).

## Run a development build

The app uses one model, **CleanMyMic Base**, stored as `hold-10k.mlpackage`.

```text
models/hold-10k.mlpackage
```

**If this package is missing, stop here.** With it available, run from the repository root:

```sh
./mac/bundle.sh
./mac/driver/build.sh
./mac/driver/install.sh
open mac/build/CleanMyMic.app
```

The driver build downloads pinned BlackHole source. Installation asks for an administrator password and briefly restarts Mac audio; finish calls and recordings first. The app uses Apple Development or ad-hoc signing, not public notarization. If macOS blocks it, review Privacy & Security rather than disabling Gatekeeper or stripping quarantine.

## First use

1. Open CleanMyMic. **Using** shows the physical microphone chosen automatically. Click it to select MacBook, AirPods, or another mic yourself.
2. In Meet, Zoom, or another call app, select **CleanMyMic** as the microphone. Keep that selection whether CleanMyMic is switched on or off. CleanMyMic manages **macOS Sound → Input** automatically.
3. Allow microphone access when macOS asks, then use your call app's microphone test. **Ready** means no app is using the virtual microphone; **In use** means audio is flowing.

The switch at the top right of the panel is the main control: **on** sends cleaned audio, **off** passes your original voice through; it does not mute or disconnect the microphone. **Settings…** at the bottom of the panel opens the Settings window, with cleanup strength and the model. **Mic check** plays your mic to your headphones so you can compare **Original** and **Cleaned** before a call; it does not change what calls get. Automatic follows the physical mic selected by macOS and falls back to an available mic if that device disconnects; a manual choice takes priority while available.

When the last app stops using the virtual microphone, CleanMyMic releases your physical mic and returns to **Ready**. macOS shows its microphone indicator during capture, including when CleanMyMic is switched off. An app's mute button may leave its microphone open; end the call or quit CleanMyMic to release it. Quitting restores a physical macOS input, but apps explicitly using CleanMyMic must switch to a physical mic if you want to keep talking.

The app registers to open at login. You can manage that in **System Settings → General → Login Items**. Audio enhancement runs locally; it does not need a cloud account or API key.

## Voice cuts out

Intermittent voice loss has been reported with AirPods Pro in a noisy café on Google Meet, while background audio remained audible. It is not fixed or isolated to a specific processing stage. For important calls, select your physical microphone directly in the call app to bypass CleanMyMic.

For a short diagnostic test, keep the physical mic unchanged and select **CleanMyMic** in Meet. Turn Meet's **Settings → Audio → Noise cancellation** off to isolate its additional processing, then compare CleanMyMic switched off and on. Off passes original audio; it is not mute. If speech disappears only with cleanup on, report that result. If it also disappears with cleanup off, compare the physical microphone directly. This is a diagnostic comparison, not a verified fix. [Meet's audio processing](https://support.google.com/meet/answer/14441737)

When reporting a failure, include the app version, physical microphone, call-app input, strength, other noise-cancellation settings, and whether bypass restored speech. Share a short audio example only if you have permission; recordings are not collected automatically.

## Remove an installer build

Quit CleanMyMic, then run in Terminal:

```sh
sudo "/Library/Application Support/CleanMyMic/uninstall.sh"
```

Restart your Mac and select a physical microphone in your call apps. This removes the app, driver, model, supporting files, and installer receipt. Preferences are kept. Disable any remaining CleanMyMic entry in Login Items.

## Remove the development build

Quit CleanMyMic. In the source checkout, run:

```sh
./mac/driver/install.sh remove
```

This asks for administrator access and restarts Mac audio. Remove the app from `mac/build/` or Applications, wherever you placed it, and disable any remaining login item. Select your physical microphone again in your call app.

## Instructions for coding agents

Use this file as the installation guide; `AGENTS.md` also governs repository changes.

- Check compatibility, repository access, and actual release assets. Report missing models or installers; do not invent download commands, substitute models, or treat source archives as installers.
- Build only for users requesting developer setup. Preserve existing checkout changes and verify the Base package before bundling or installing the driver.
- Explain the required restart for the installer, or the immediate audio restart for the development driver script. Let users enter credentials and grant permissions through macOS; do not bypass security controls. Do not present the unsigned private candidate as a trusted public download.
- Verify the app, virtual microphone, and audio in the user's call app. Report untested steps; compilation alone does not verify microphone cleanup.
