# Release preparation

**v0.1.0-alpha.1 — First preview** is an experimental public preview. Review the [release notes](releases/v0.1.0-alpha.1.md).

App version: `0.1.0`, build `1`. Python version: `0.1.0a1`. Installer receipt version: `0.1.0.1`. The installer includes CleanMyMic Base's compiled Core ML model. Training checkpoints and Core ML source packages remain outside Git.

## Readiness

**November 2, 2026: public experimental preview.** The source and a signed installer are public; the app is not yet dependable for everyday calls.

| Area | Current state | Needed before release |
|---|---|---|
| Call reliability | Café/AirPods/Meet voice loss reported; cause unresolved | Reproduce and fix before an everyday-use app release; prominently disclose for any source preview |
| Models | Base is bundled in the installer, with its weight hash in `build.json` | Verify the same model is in the final signed artifact |
| Installation | Signed, notarized `.pkg`; payload checks pass; source/notices included | Clean-Mac install/upgrade/removal tests |
| Public contents | The public repository starts from a fresh history; development history stays private | — |
| Review protection | Rulesets active on `main` and `v*` tags; outside PRs need approval before CI runs | — |

The source and routing CI passes on the implementation commit, and local checks cover bypass/resume and capture starting/stopping for virtual-mic consumers. They do not establish real Meet/Zoom reliability, Bluetooth recovery, battery use, or end-to-end latency. [Benchmark scope](benchmarks.md) and the [known failure](../INSTALL.md#voice-cuts-out) remain part of release review.

## Review and merge

Changes go through draft PRs; the owner reviews and merges manually. Auto-merge is off. Since November 2, 2026, the rulesets below enforce this.

The [main ruleset](../.github/main-ruleset.json) requires a PR, code-owner approval, resolved conversations, and passing `Source checks` on an up-to-date branch; it blocks direct pushes, force pushes, and deletion. Its PR-only bypass lets `thisisvaze` (user ID `11044960`) manually merge owner-authored PRs, since authors cannot approve themselves. It cannot distinguish the owner from tools using the owner's credentials.

The [tag ruleset](../.github/tag-ruleset.json) blocks deleting or moving `v*` release tags; the owner can bypass it to repoint a tag at a rebuilt release. Workflows from outside contributors' PRs wait for owner approval, and private vulnerability reporting is on.

Both rulesets were created from these files on November 2, 2026. To recreate one, or to check what is enforced:

```sh
gh api --method POST repos/thisisvaze/cleanmymic/rulesets --input .github/main-ruleset.json
gh api repos/thisisvaze/cleanmymic/rules/branches/main
```

Check code-owner recognition and CI on a PR before relying on enforcement. See GitHub's [ruleset API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset) and [review limitations](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews).

## Publish a release

1. After merge and passing CI, point the release at the reviewed `main` commit.
2. Keep the unresolved-failure disclosure and describe accurately which artifacts are available and tested. Attach only a signed, notarized package. Copy the [release notes](releases/v0.1.0-alpha.1.md), resolving relative links against that exact commit. No workflow publishes automatically.

## Build the installer

With Xcode and `models/hold-10k.mlpackage` available locally, build an unsigned test candidate:

```sh
./mac/installer/build.sh --unsigned
```

Outputs go to `mac/build/release/`: one `.pkg`, the corresponding driver source archive, and `SHA256SUMS-unsigned.txt`. The installer contains the app, the HAL driver, and Base; supporting files go in `/Library/Application Support/CleanMyMic`. It requires Apple Silicon, macOS 15+, administrator access, and a restart. Native Installer handles replacement, closes the app, and keeps preferences. It runs no custom privileged installation scripts.

For a public candidate, install **Developer ID Application** and **Developer ID Installer** certificates for the chosen Apple team, and store notarization credentials in Keychain with `xcrun notarytool store-credentials CleanMyMic`. Enter credentials locally; do not put passwords in scripts, Git, or chat.

```sh
APPLICATION_IDENTITY='Developer ID Application: Aaditya Vaze (TEAM_ID)' \
INSTALLER_IDENTITY='Developer ID Installer: Aaditya Vaze (TEAM_ID)' \
NOTARY_PROFILE=CleanMyMic ./mac/installer/build.sh
```

The default build requires these settings; it never silently falls back to unsigned output. It signs the app with hardened runtime and microphone entitlement, signs the driver and package, submits to Apple, checks acceptance, staples the ticket, and runs Gatekeeper assessment. The resulting name omits `-unsigned`; checksums are written after stapling. [Apple distribution signing](https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution).

Current signing setup: Developer ID Application and Installer certificates for Aaditya Vaze (`2FN5ZL5G56`) and the `CleanMyMic` notarization profile are set up locally.

The build runs `python3 mac/installer/verify.py <package.pkg>` to expand the actual package and check destinations, architecture, code-signature integrity, Base weights, driver source, notices, and restart requirements. The Base weight SHA-256 is `a8896e66f673dd54b7251577806d51685d2def46c918310c73a95ca6319373bc`, matching the [measured model](benchmarks.md). The embedded `build.json` records the source commit and whether it had uncommitted changes. CI checks installer configuration without needing the model.

Before attaching a final signed artifact, test a clean supported Mac: install, restart, microphone permission, received call audio with removal on/off, idle capture release, upgrade, and uninstall/restart. These checks need real hardware or a suitable clean VM; expanding a package is not installation. No clean-Mac install test has been completed yet.
