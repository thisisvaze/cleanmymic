#!/bin/bash
# One offline installer. Default: Developer ID signing + notarization; --unsigned: private testing only.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  '')
    : "${APPLICATION_IDENTITY:?Set Developer ID Application identity}"
    : "${INSTALLER_IDENTITY:?Set Developer ID Installer identity}"
    : "${NOTARY_PROFILE:?Set the notarytool keychain profile}"
    [[ "$APPLICATION_IDENTITY" == 'Developer ID Application: '* ]]
    [[ "$INSTALLER_IDENTITY" == 'Developer ID Installer: '* ]]
    SUFFIX=''
    ;;
  --unsigned) SUFFIX=-unsigned ;;
  *) echo "Usage: $0 [--unsigned]" >&2; exit 2 ;;
esac
python3 mac/installer/verify.py
./mac/bundle.sh
./mac/driver/build.sh
VERSION=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' mac/Info.plist)
BUILD=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' mac/Info.plist)
RELEASE="$VERSION-alpha.$BUILD"
OUT="$PWD/mac/build/release"
mkdir -p "$OUT"
WORK=$(mktemp -d "$PWD/mac/build/installer.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
ROOT="$WORK/root"
SUPPORT="$ROOT/Library/Application Support/CleanMyMic"
APP="$ROOT/Applications/CleanMyMic.app"
DRIVER="$ROOT/Library/Audio/Plug-Ins/HAL/CleanMyMic.driver"
mkdir -p "$(dirname "$APP")" "$(dirname "$DRIVER")" "$SUPPORT/licenses"
ditto mac/build/CleanMyMic.app "$APP"
ditto mac/driver/build/CleanMyMic.driver "$DRIVER"
cp LICENSE THIRD_PARTY_NOTICES.md "$SUPPORT/"
cp licenses/*.txt "$SUPPORT/licenses/"
cp mac/installer/uninstall.sh "$SUPPORT/"
chmod 755 "$SUPPORT/uninstall.sh"

# Include the exact driver sources and build recipe, without local Git metadata or build caches.
SOURCE="$WORK/CleanMyMic-driver-source"
mkdir -p "$SOURCE/mac/driver/build/BlackHole"
cp mac/driver/build.sh "$SOURCE/mac/driver/"
cp mac/AppIcon.icns "$SOURCE/mac/"
(cd mac/driver/build/BlackHole; git ls-files -z | tar --null -T - -cf -) |
  tar -xf - -C "$SOURCE/mac/driver/build/BlackHole"
cp licenses/BlackHole.txt "$SOURCE/LICENSE.txt"
printf 'CleanMyMic virtual microphone source\n\nBlackHole revision: %s\n\nWith Xcode installed, run ./mac/driver/build.sh from this directory.\nThe vendored source needs no download. Output: mac/driver/build/CleanMyMic.driver.\n' \
  "$(git -C mac/driver/build/BlackHole rev-parse HEAD)" > "$SOURCE/README.txt"
SOURCE_ARCHIVE="CleanMyMic-driver-source-$RELEASE.tar.gz"
COPYFILE_DISABLE=1 tar -czf "$WORK/$SOURCE_ARCHIVE" -C "$WORK" CleanMyMic-driver-source
cp "$WORK/$SOURCE_ARCHIVE" "$SUPPORT/driver-source.tar.gz"
python3 - "$SUPPORT/build.json" <<'PY'
import hashlib, json, pathlib, subprocess, sys
def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()
weights = pathlib.Path('models/hold-10k.mlpackage/Data/com.apple.CoreML/weights/weight.bin')
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    'source_commit': git('rev-parse', 'HEAD'),
    'source_dirty': bool(git('status', '--porcelain')),
    'model': 'CleanMyMic Base',
    'model_weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
    'driver_revision': git('-C', 'mac/driver/build/BlackHole', 'rev-parse', 'HEAD'),
}, indent=2) + '\n')
PY
if [[ -n "$SUFFIX" ]]; then
  codesign --force --sign - --options runtime --timestamp=none --entitlements mac/installer/entitlements.plist "$APP"
  codesign --force --sign - --timestamp=none "$DRIVER"
else
  codesign --force --sign "$APPLICATION_IDENTITY" --options runtime --timestamp --entitlements mac/installer/entitlements.plist "$APP"
  codesign --force --sign "$APPLICATION_IDENTITY" --options runtime --timestamp "$DRIVER"
fi
pkgbuild --root "$ROOT" --install-location / --ownership recommended \
  --identifier dev.vaze.CleanMyMic.pkg --version "$VERSION.$BUILD" \
  --component-plist mac/installer/components.plist "$WORK/CleanMyMic-component.pkg"
PKG="$WORK/CleanMyMic-$RELEASE-arm64$SUFFIX.pkg"
set --
if [[ -z "$SUFFIX" ]]; then set -- --sign "$INSTALLER_IDENTITY" --timestamp; fi
productbuild --distribution mac/installer/Distribution.xml --resources mac/installer/resources \
  --package-path "$WORK" "$@" "$PKG"
python3 mac/installer/verify.py "$PKG"
if [[ -z "$SUFFIX" ]]; then
  xcrun notarytool submit "$PKG" --keychain-profile "$NOTARY_PROFILE" --wait --output-format plist > "$WORK/notary.plist"
  test "$(/usr/libexec/PlistBuddy -c 'Print :status' "$WORK/notary.plist")" = Accepted || {
    cat "$WORK/notary.plist" >&2; exit 1;
  }
  xcrun stapler staple "$PKG"
  xcrun stapler validate "$PKG"
  spctl --assess --type install --verbose=2 "$PKG"
fi
pkgutil --check-signature "$PKG" || { [[ -n "$SUFFIX" ]] || exit 1; }
(cd "$WORK"; shasum -a 256 "$(basename "$PKG")" "$SOURCE_ARCHIVE" > "SHA256SUMS$SUFFIX.txt")
# Keep failed or unnotarized release builds out of the downloadable output directory.
mv "$PKG" "$WORK/$SOURCE_ARCHIVE" "$WORK/SHA256SUMS$SUFFIX.txt" "$OUT/"
printf '\nInstaller: %s/%s\n' "$OUT" "$(basename "$PKG")"
