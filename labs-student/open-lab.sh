#!/bin/bash
# Opens the page of the lab (the desktop icon "Myduino Jetson Lab" runs this) in a browser that really works on this Jetson.
# xdg-open alone does nothing when the default browser (often Firefox) is not installed. A snap browser (Brave, Chromium, Firefox of Ubuntu 22.04)
# can fail on a Jetson image without AppArmor: so every browser is tried in turn, a normal (deb) one first, and the next one is tried when it fails.
URL="http://localhost:8870"
export PATH="$PATH:/snap/bin"
export SNAP_REEXEC=0        # the kernel of some Jetsons cannot run the newer snap helper: snap browsers fail without this

# try <command...>: true when the browser started (it is still running after 8 seconds, or it ended well)
try() {
  "$@" > /dev/null 2>&1 &
  local pid=$!
  for _ in 1 2 3 4 5 6 7 8; do
    sleep 1
    kill -0 "$pid" 2> /dev/null || { wait "$pid"; return $?; }
  done
  return 0
}

for b in epiphany epiphany-browser google-chrome google-chrome-stable chromium chromium-browser firefox brave brave-browser; do
  command -v "$b" > /dev/null || continue
  try "$b" "$URL" && exit 0
done
try xdg-open "$URL" && exit 0
echo "No browser could open $URL. Install one (sudo apt install epiphany-browser) and double-click the icon again." >&2
exit 1
