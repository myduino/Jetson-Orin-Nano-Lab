#!/bin/bash
# Run ONCE on each Jetson after cloning this folder:
#     bash labs-setup.sh <teacher address:port> [name of this Jetson]
# Give each Jetson a different name (for example jetson-01). Without one the name is made from the host name and a short
# code of this Jetson, because Jetsons flashed from the same image have the same host name.
# The teacher page prints the exact line (for example:  bash labs-setup.sh 192.168.0.10:8800).
# Needs internet. Safe to run again. It installs the lab agent. The agent starts by itself whenever the Jetson starts and
# reports to the teacher page. When the teacher puts a student on this Jetson, the agent makes that student's own Python
# environment (JupyterLab, 2 to 3 minutes) in the student's folder and starts their JupyterLab.
# The rest of the setup (CuPy, PyTorch, Ultralytics) is done by each student in the Lab 0 notebook, where they can watch it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Solo Lab (no teacher app): bash labs-setup.sh --solo <package> [name]   packages: foundations, vision, applied, full (labs-student/packages.json)
# The desktop icon then gives the student at this Jetson their own lab when no teacher app answers. --solo off turns it off.
SOLO=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --solo) SOLO="${2:-}"; [ -n "$SOLO" ] || { echo "Say which package: --solo vision"; exit 1; }; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
TEACHER="${ARGS[0]:-}"; NAME="${ARGS[1]:-}"
# With --solo, a single word that is not an address is the name of this Jetson (not a teacher address).
if [ -n "$SOLO" ] && [ ${#ARGS[@]} -eq 1 ] && ! [[ "${ARGS[0]}" =~ ^[0-9.]+(:[0-9]+)?$ || "${ARGS[0]}" == *:* ]]; then NAME="${ARGS[0]}"; TEACHER=""; fi
if [ -n "$SOLO" ] && [ "$SOLO" != "off" ] && ! python3 -c "import json,sys;sys.exit(0 if any(p['id']==sys.argv[1] for p in json.load(open(sys.argv[2]))) else 1)" "$SOLO" "$HERE/labs-student/packages.json"; then
  echo "Unknown package '$SOLO'. Use one of: $(python3 -c "import json,sys;print(', '.join(p['id'] for p in json.load(open(sys.argv[1]))))" "$HERE/labs-student/packages.json")"; exit 1
fi
[ -n "$TEACHER" ] || [ -n "$SOLO" ] || read -r -p "Teacher page address (for example 192.168.0.10:8800): " TEACHER
[ -z "$TEACHER" ] || case "$TEACHER" in *:*) ;; *) TEACHER="$TEACHER:8800" ;; esac
if [ -z "$NAME" ]; then NAME="$(hostname | tr 'A-Z-' 'a-z_')_$(cut -c1-4 /etc/machine-id 2>/dev/null)"; fi
NAME="$(printf '%s' "$NAME" | tr 'A-Z-' 'a-z_' | sed 's/[^a-z0-9_]/_/g' | cut -c1-20)"

echo "== Myduino Jetson Orin Nano Super Lab: setup of this Jetson =="
echo "== 1. Python tools =="
sudo apt update
sudo apt install -y python3-venv python3-pip git curl mosquitto mosquitto-clients tesseract-ocr libzbar0 alsa-utils espeak-ng
# The agent runs its own MQTT broker for the students (ESP32 labs, phone apps): the one of the system must not take the port.
sudo systemctl disable --now mosquitto > /dev/null 2>&1 || true
# jtop (used in Lab 1) needs its own service, or it says "jtop service is not active".
sudo -H /usr/bin/pip3 install -U jetson-stats && sudo jtop --install-service

echo "== 2. The lab folder =="
# Only two folders of the lab live in the home folder: this one, and ~/myduino-jetson-lab (the students and the settings).
# Older versions of this script wrote an alias into ~/.bashrc and a settings file in the home folder: remove them.
sed -i '/^alias lab=/d;/^alias jet=/d' ~/.bashrc 2>/dev/null
rm -f ~/.lab_agent.json
mkdir -p ~/myduino-jetson-lab

# The language model engine is only for Labs 20 to 27. A Jetson with a small package (Foundations or Vision AI) and no teacher app does not
# need it, which saves about 20 minutes. With a teacher app, or with Applied AI or Full Course, it is built (run this again to build it later).
NEED_LLM="$(python3 - "$TEACHER" "$SOLO" "$HERE/labs-student/packages.json" <<'PY'
import json, os, sys
teacher, solo, packages = sys.argv[1:4]
try:
    old = json.load(open(os.path.expanduser("~/myduino-jetson-lab/.agent.json")))
except (OSError, ValueError):
    old = {}
teacher = teacher or old.get("teacher", "")
pack = old.get("solo", {}).get("package", "") if not solo else ("" if solo == "off" else solo)
try:
    top = next((p["to"] for p in json.load(open(packages)) if p["id"] == pack), None)
except (OSError, ValueError):
    top = None
print("0" if (not teacher and top is not None and top < 20) else "1")
PY
)"
echo "== 3. Language model engine (llama.cpp, used by Labs 20 to 27) =="
if [ "$NEED_LLM" = "0" ]; then
  echo "Skipped: the package of this Jetson has no language model lab. Run this again with a bigger package to build it."
else
# Built once with CUDA for the GPU of the Orin Nano. It takes about 20 minutes. The version is fixed (v0.9.0, the one the labs were tested with):
# it has the speech endpoint /tts that Labs 22 and 23 use. GGML_CUDA_NO_VMM stops a GPU memory error of the audio models on this board.
# The model files are downloaded later by the first student, in Lab 20. Skipped when this version is already built.
LLAMA_COMMIT=cd959bae99ff4bed22bef2604f45241c09bbb8c8
if [ -x ~/llama.cpp/build/bin/llama-server ] && [ "$(cat ~/llama.cpp/.myduino_build 2>/dev/null)" = "$LLAMA_COMMIT" ]; then
  echo "llama-server is already built (v0.9.0)."
else
  sudo apt install -y cmake build-essential
  if [ ! -d ~/llama.cpp/.git ]; then git init -q ~/llama.cpp && git -C ~/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp; fi
  git -C ~/llama.cpp fetch --depth 1 origin "$LLAMA_COMMIT" && git -C ~/llama.cpp checkout -q FETCH_HEAD
  (cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j4 --target llama-server \
    && echo "$LLAMA_COMMIT" > .myduino_build) || echo "llama.cpp did not build: run this script again, or ask the teacher."
fi
fi

echo "== 4. Power mode =="
command -v nvpmodel > /dev/null && echo no | sudo nvpmodel -m 2 && echo "Power mode: MAXN SUPER (full speed after the next reboot)"

echo "== 5. The lab agent =="
# The settings keep what is already there: running this again for the teacher app does not turn the Solo Lab off, and the other way round.
python3 - "$TEACHER" "$NAME" "$SOLO" <<'PY'
import json, os, sys
path = os.path.expanduser("~/myduino-jetson-lab/.agent.json")
teacher, name, solo = sys.argv[1:4]
try:
    cfg = json.load(open(path))
except (OSError, ValueError):
    cfg = {}
cfg["id"] = name
if teacher:
    cfg["teacher"] = teacher
if solo == "off":
    cfg.pop("solo", None)
elif solo:
    cfg["solo"] = {"package": solo}
json.dump(cfg, open(path, "w"))
PY
chmod 600 ~/myduino-jetson-lab/.agent.json
sudo tee /etc/systemd/system/myduino-lab-agent.service > /dev/null <<UNIT
[Unit]
Description=Myduino lab agent (starts the JupyterLab of the students)
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$HERE
ExecStart=/usr/bin/python3 $HERE/labs-student/agent.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
# The "Myduino Jetson Lab" icon (desktop and app menu): opens the JupyterLab of the student who works at this Jetson.
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo ~/Desktop)"
mkdir -p ~/.local/share/applications "$DESKTOP_DIR"
rm -f "$DESKTOP_DIR/Myduino Lab.desktop"                       # the first name of the icon
cat > ~/.local/share/applications/myduino-lab.desktop <<ICON
[Desktop Entry]
Type=Application
Name=Myduino Jetson Lab
Comment=Open your JupyterLab
Exec=bash "$HERE/labs-student/open-lab.sh"
Icon=$HERE/labs-student/myduino-icon.png
Terminal=false
Categories=Education;
ICON
cp ~/.local/share/applications/myduino-lab.desktop "$DESKTOP_DIR/Myduino Jetson Lab.desktop"
chmod +x "$DESKTOP_DIR/Myduino Jetson Lab.desktop"
# The desktop only starts an icon marked "trusted". This script may run over SSH, where there is no desktop session, so use a private one.
ICONFILE="$DESKTOP_DIR/Myduino Jetson Lab.desktop"
{ dbus-run-session -- gio set "$ICONFILE" metadata::trusted true || gio set "$ICONFILE" metadata::trusted true; } 2>/dev/null \
  || echo "Could not mark the icon as trusted. On the Jetson: right-click the icon and choose Allow Launching."

sudo systemctl daemon-reload
sudo systemctl enable myduino-lab-agent > /dev/null 2>&1
sudo systemctl restart myduino-lab-agent
sleep 3
# What the settings say now (running this again keeps the teacher address or the Solo Lab that were set before).
TEACHER="$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/myduino-jetson-lab/.agent.json'))).get('teacher',''))")"
SOLO_ON="$(python3 -c "import json,os;print((json.load(open(os.path.expanduser('~/myduino-jetson-lab/.agent.json'))).get('solo') or {}).get('package',''))")"
if [ -n "$TEACHER" ]; then
  if curl -s -m 5 "http://$TEACHER/" > /dev/null; then
    echo "The teacher page at $TEACHER answers. Refresh the teacher page: this Jetson should be in the list."
  else
    echo "Cannot reach the teacher page at $TEACHER. Check that the teacher PC is on, on the same network,"
    echo "and that Windows Firewall allows port 8800. The agent keeps trying by itself."
  fi
fi

echo
if [ -n "$SOLO_ON" ]; then
  echo "Solo Lab is on (package: $SOLO_ON). The student double-clicks Myduino Jetson Lab on the desktop of this Jetson:"
  echo "when the teacher has no lab for this Jetson, it makes their folder and opens JupyterLab (a few minutes the first time)."
fi
if [ -n "$TEACHER" ]; then
  echo "Next: the teacher puts a student on this Jetson, and the student opens the address from the teacher page."
fi
echo "Each student then runs Lab 0: Part 0 (system programs, once for this Jetson) and Part 0b (their own AI packages and models)."
