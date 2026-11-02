#!/bin/bash
# Build (once) and run the live test tool. Usage in live/main.swift.
here="$(cd "$(dirname "$0")" && pwd)"
src=("$here/live/main.swift" "$here/Sources/CleanMyMic/Enhancer.swift")
[ "$here/.live" -nt "${src[0]}" ] && [ "$here/.live" -nt "${src[1]}" ] ||
  swiftc -O -suppress-warnings "${src[@]}" -o "$here/.live" || exit 1
exec "$here/.live" "$@"
