#!/bin/bash
# Opens the page of the lab (the desktop icon "Myduino Jetson Lab" runs this) in a browser that is really installed.
# xdg-open alone does nothing when the default browser (often Firefox) is not installed on the Jetson.
URL="http://localhost:8870"
export PATH="$PATH:/snap/bin"
export SNAP_REEXEC=0        # the kernel of the Jetson cannot run the newer snap helper: snap browsers (Brave, Firefox) fail without this
DEF="$(xdg-settings get default-web-browser 2>/dev/null)"
if [ -n "$DEF" ] && { [ -f "/usr/share/applications/$DEF" ] || [ -f "$HOME/.local/share/applications/$DEF" ] || [ -f "/var/lib/snapd/desktop/applications/$DEF" ]; }; then
  exec xdg-open "$URL"
fi
for b in firefox brave chromium-browser chromium google-chrome epiphany; do
  command -v "$b" > /dev/null && exec "$b" "$URL"
done
exec xdg-open "$URL"
