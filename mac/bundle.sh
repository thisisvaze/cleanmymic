#!/bin/sh
# ./bundle.sh          -> build/CleanMyMic.app (signed with your Apple Development cert so the mic permission survives
#                         rebuilds; ad-hoc if none)
# ./bundle.sh install  -> also replaces /Applications/CleanMyMic.app and relaunches it
# CleanMyMic Base: ../models/hold-10k.mlpackage, from train/export.py.
set -e
cd "$(dirname "$0")"
test -d ../models/hold-10k.mlpackage || { echo "Missing CleanMyMic Base: models/hold-10k.mlpackage" >&2; exit 1; }
swift build -c release
APP=build/CleanMyMic.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/CleanMyMic "$APP/Contents/MacOS/"
cp Info.plist "$APP/Contents/"
cp AppIcon.icns "$APP/Contents/Resources/"
xcrun coremlcompiler compile ../models/hold-10k.mlpackage build/ > /dev/null
mv build/hold-10k.mlmodelc "$APP/Contents/Resources/Hold-10k.mlmodelc"
ID=$(security find-identity -v -p codesigning | awk '/Apple Development/ {print $2; exit}')
codesign --force --sign "${ID:--}" "$APP"
if [ "$1" = install ]; then
  pkill -x CleanMyMic || true
  rm -rf /Applications/CleanMyMic.app
  cp -R "$APP" /Applications/
  open /Applications/CleanMyMic.app
  echo "/Applications/CleanMyMic.app"
else
  echo "$APP"
fi
