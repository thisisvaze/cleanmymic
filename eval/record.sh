#!/bin/bash
# Record one take on several Mac mics at once. Press q (or Enter) to stop.
#   eval/record.sh quiet                                  # MacBook Pro + Studio Display XDR mics
#   eval/record.sh echo-kitchen "MacBook Pro Microphone"  # one mic only
here="$(cd "$(dirname "$0")" && pwd)"
[ "$here/.record" -nt "$here/record.swift" ] || swiftc -O -suppress-warnings "$here/record.swift" -o "$here/.record" || exit 1
exec "$here/.record" "$@"
