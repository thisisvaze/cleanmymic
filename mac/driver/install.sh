#!/bin/sh
# Installs the virtual mic for all users. Asks for your password; Core Audio restarts, so all sound stops for a moment.
# To remove it: ./install.sh remove
cd "$(dirname "$0")"
HAL=/Library/Audio/Plug-Ins/HAL
if [ "$1" = remove ]; then
  CMD="rm -rf $HAL/CleanMyMic.driver"
else
  [ -d build/CleanMyMic.driver ] || { echo "run ./build.sh first"; exit 1; }
  TMP=$(mktemp -d /tmp/cleanmymic.XXXXXX)  # root can't read ~/Documents (macOS privacy), but this user can copy out of it
  cp -R build/CleanMyMic.driver "$TMP/"
  CMD="rm -rf $HAL/CleanMyMic.driver && cp -R $TMP/CleanMyMic.driver $HAL/ && chown -R root:wheel $HAL/CleanMyMic.driver"
fi
osascript -e "do shell script \"$CMD && killall -9 coreaudiod\" with administrator privileges"
