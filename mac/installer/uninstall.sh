#!/bin/sh
# Removes this install's files; leaves preferences and other audio drivers intact.
set -eu
test "$(/usr/bin/id -u)" = 0 || { echo 'Run this script with sudo.' >&2; exit 1; }
if /usr/bin/pgrep -x CleanMyMic >/dev/null; then
  echo 'Quit CleanMyMic before removing it.' >&2; exit 1
fi
APP=/Applications/CleanMyMic.app
DRIVER=/Library/Audio/Plug-Ins/HAL/CleanMyMic.driver
SUPPORT='/Library/Application Support/CleanMyMic'
for path in "$APP" "$DRIVER" "$SUPPORT"; do
  test ! -L "$path" || { echo "Refusing to remove a symbolic link: $path" >&2; exit 1; }
done
for bundle in "$APP" "$DRIVER"; do
  if [ -e "$bundle" ]; then
    expected=dev.vaze.CleanMyMic
    test "$bundle" != "$DRIVER" || expected=dev.vaze.CleanMyMic.driver
    actual=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$bundle/Contents/Info.plist")
    test "$actual" = "$expected" || { echo "Unexpected app or driver at $bundle" >&2; exit 1; }
  fi
done
/bin/rm -rf -- "$APP" "$DRIVER"
/bin/rm -f -- "$SUPPORT/LICENSE" "$SUPPORT/THIRD_PARTY_NOTICES.md" "$SUPPORT/build.json" \
  "$SUPPORT/driver-source.tar.gz" "$SUPPORT/licenses/BlackHole.txt" "$SUPPORT/licenses/DeepFilterNet-MIT.txt" "$SUPPORT/uninstall.sh"
/bin/rmdir "$SUPPORT/licenses" "$SUPPORT" 2>/dev/null || true
if /usr/sbin/pkgutil --pkg-info dev.vaze.CleanMyMic.pkg >/dev/null 2>&1; then
  /usr/sbin/pkgutil --forget dev.vaze.CleanMyMic.pkg
fi
echo 'Removed CleanMyMic. Restart your Mac, then select a physical microphone in your call apps.'
