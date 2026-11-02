#!/bin/sh
# Builds the CleanMyMic virtual mic: BlackHole (GPL-3.0, (c) Existential Audio, github.com/ExistentialAudio/BlackHole)
# under our name, at a pinned commit. A separate program from the app (MIT), which only talks to it through Core Audio.
# ./build.sh -> build/CleanMyMic.driver, then ./install.sh
#
# What differs from BlackHole, all below:
#   a mic called CleanMyMic with no output, and a hidden output behind it that only the app writes to
#   48 kHz only, no volume control (call apps ride mic volume)
#   it reports USB, not virtual: some apps hide virtual mics (eqMac does the same)
set -e
cd "$(dirname "$0")"
REV=574029f0d3175ccf29a826d74be4da3967b1cb92
mkdir -p build
cd build
if [ ! -d BlackHole ]; then
  git init -q BlackHole
  git -C BlackHole fetch -q --depth 1 https://github.com/ExistentialAudio/BlackHole $REV
  git -C BlackHole checkout -q FETCH_HEAD
fi
cd BlackHole
# Release source archives contain the same tree without Git metadata.
if [ -d .git ]; then
  test "$(git rev-parse HEAD)" = "$REV" || { echo "Unexpected BlackHole revision" >&2; exit 1; }
  git checkout -q -- .
fi
sed -i '' 's/= kAudioDeviceTransportTypeVirtual;/= kAudioDeviceTransportTypeUSB;/' BlackHole/BlackHole.c
cp ../../../AppIcon.icns BlackHole/BlackHole.icns
xcodebuild -project BlackHole.xcodeproj -target BlackHole -configuration Release -quiet \
  CONFIGURATION_BUILD_DIR="$PWD/../out" PRODUCT_NAME=CleanMyMic PRODUCT_BUNDLE_IDENTIFIER=dev.vaze.CleanMyMic.driver \
  CODE_SIGNING_ALLOWED=NO GCC_PREPROCESSOR_DEFINITIONS='$GCC_PREPROCESSOR_DEFINITIONS
  kDriver_Name=\"CleanMyMic\" kPlugIn_BundleID=\"dev.vaze.CleanMyMic.driver\" kManufacturer_Name=\"CleanMyMic\"
  kHas_Driver_Name_Format=false kDevice_Name=\"CleanMyMic\" kDevice2_Name=\"CleanMyMicCable\"
  kDevice_HasOutput=false kDevice2_HasInput=false kDevice2_HasOutput=true kDevice2_IsHidden=true
  kSampleRates=48000 kEnableVolumeControl=false'
cd ..
# its own plug-in id, as BlackHole's installer does, so it can sit next to BlackHole
sed -i '' 's/e395c745-4eea-4d94-bb92-46224221047c/3dacaa64-832f-4c78-b018-5d1886119242/' out/CleanMyMic.driver/Contents/Info.plist
ID=$(security find-identity -v -p codesigning | awk '/Apple Development/ {print $2; exit}')
codesign --force --sign "${ID:--}" out/CleanMyMic.driver
rm -rf CleanMyMic.driver
mv out/CleanMyMic.driver .
echo "$PWD/CleanMyMic.driver"
