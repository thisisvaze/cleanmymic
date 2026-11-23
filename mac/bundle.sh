#!/bin/sh
# ./bundle.sh          -> build/CleanMyMic.app (signed with Developer ID or Apple Development cert so mic permission survives rebuilds; ad-hoc if none)
# ./bundle.sh install  -> also replaces /Applications/CleanMyMic.app and relaunches it
# Model: ../models/CleanMyMicStudio.mlpackage (or override with MODEL=/path/to/model.mlpackage).
set -e
cd "$(dirname "$0")"
MODEL="${MODEL:-../models/CleanMyMicStudio.mlpackage}"
test -d "$MODEL" || { echo "Missing Core ML model package: $MODEL" >&2; exit 1; }
swift build -c release
APP=build/CleanMyMic.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/CleanMyMic "$APP/Contents/MacOS/"
cp Info.plist "$APP/Contents/"
cp AppIcon.icns "$APP/Contents/Resources/"
TMP=$(mktemp -d build/model.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
xcrun coremlcompiler compile "$MODEL" "$TMP" > /dev/null
COMPILED=$(find "$TMP" -maxdepth 1 -name '*.mlmodelc' | head -n 1)
mv "$COMPILED" "$APP/Contents/Resources/CleanMyMicStudio.mlmodelc"
ID=$(security find-identity -v -p codesigning | awk '/Developer ID Application/ {print $2; exit}')
if [ -n "$ID" ]; then
  codesign --force --options runtime --timestamp --entitlements installer/entitlements.plist --sign "$ID" "$APP"
else
  ID=$(security find-identity -v -p codesigning | awk '/Apple Development/ {print $2; exit}')
  codesign --force --entitlements installer/entitlements.plist --sign "${ID:--}" "$APP"
fi
if [ "$1" = install ]; then
  pkill -x CleanMyMic || true
  rm -rf /Applications/CleanMyMic.app
  cp -R "$APP" /Applications/
  open /Applications/CleanMyMic.app
  echo "/Applications/CleanMyMic.app"
else
  echo "$APP"
fi
