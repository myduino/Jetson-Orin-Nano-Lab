"""The Jetson agent of the Myduino lab. It runs on every Jetson as a service (labs-setup.sh installs it).

Every few seconds it tells the teacher server which Jetson this is, its IP address and its cameras, and asks which
students should work here. For each student it makes the folder ~/myduino-jetson-lab/<class>/<student> and starts
that student's JupyterLab on the port and with the PIN the teacher page shows, with the chosen camera. When the
teacher takes a student away, the agent stops that JupyterLab. Each student has their own Python environment in
their folder (.venv), made by the agent the first time. Only the Python standard library is used.

    python3 labs-student/agent.py            (the service does this)
"""
import base64
import glob
import html
import http.server
import hashlib
import json
import os
import re
import shutil
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # the course folder
HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "myduino-jetson-lab")                          # everything of the lab: ROOT/<class>/<student>, ROOT/.agent.json
CONFIG = os.environ.get("LAB_AGENT_CONFIG", os.path.join(ROOT, ".agent.json"))
SYSFS = os.environ.get("LAB_SYSFS", "/sys/class/video4linux")
DEV = os.environ.get("LAB_DEV", "/dev")
MAX_LABS = 45                                                                     # lab0 ... lab44 folders in each student's folder
NAME = re.compile(r"^[a-z0-9_]{1,20}$")
GRACE = 15 * 60                                                         # seconds without the teacher before every student session is closed
LAUNCH_PORT = 8870                                                      # the desktop icon opens http://localhost:8870 (this Jetson only)
EVERY = 5                                                               # seconds between two reports
SOLO_KEY, SOLO_PORT = "solo__student", 8880                             # Solo Lab: the one student who works at this Jetson without a teacher app

running = {}          # key -> {"proc", "port", "pin", "camera", "started"}
preparing = {}        # key -> time when the student's Python environment is being made
failed = {}           # key -> (time, message) of the last failed attempt to make it
deleted = []          # keys of folders deleted since the last report
renamed = []          # "old>new" of class folders renamed since the last report
rename_failed = {}    # "old>new" -> message, shown to the teacher until it works
CFG = {}              # the settings of this Jetson (.agent.json)
last_plan = []        # the last plan of the teacher app


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- the Jetson
def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def cameras():
    """USB cameras in order: the capture node of each camera (sysfs index 0), numbered 0, 1, 2."""
    found = []
    try:
        names = sorted((d for d in os.listdir(SYSFS) if d.startswith("video")), key=lambda d: int(d[5:]))
    except OSError:
        return found
    for d in names:
        if read(os.path.join(SYSFS, d, "index")) == "0":
            found.append({"number": len(found), "device": os.path.join(DEV, d), "name": read(os.path.join(SYSFS, d, "name")) or "?"})
    return found


def my_ip(teacher_host):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((teacher_host, 9))                     # sends nothing: finds the address that reaches the teacher PC
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def venv_of(home):
    return os.path.join(home, ".venv")


def venv_ready(home):
    return os.path.exists(os.path.join(venv_of(home), ".ready"))


def make_venv(key, home):
    """Make the student's own Python environment (about 2 to 3 minutes). Runs in a thread, so other students are not held up."""
    venv = venv_of(home)
    os.makedirs(os.path.join(home, ".jupyter"), exist_ok=True)
    with open(os.path.join(home, ".jupyter", "prepare.log"), "ab") as out:
        env = student_env(home, os.path.basename(os.path.dirname(home)), "")     # pip and python write only inside the folder

        def run(*cmd):
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True, env=env, cwd=home)
        try:
            log("making the Python environment of", key.replace("__", "/"))
            shutil.rmtree(venv, ignore_errors=True)
            run("/usr/bin/python3", "-m", "venv", "--system-site-packages", venv)
            run(os.path.join(venv, "bin", "pip"), "install", "--upgrade", "pip")
            run(os.path.join(venv, "bin", "pip"), "install", "numpy<2", "jupyterlab", "ipywidgets", "openpyxl", "paho-mqtt")
            site = subprocess.run([os.path.join(venv, "bin", "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                  capture_output=True, text=True, check=True, env=env).stdout.strip()
            # OpenCV on the Jetson makes NumPy print a harmless warning: hide only that one message.
            with open(os.path.join(site, "myduino_quiet.pth"), "w") as f:
                f.write('import warnings; warnings.filterwarnings("ignore", message="The value of the smallest subnormal")\n')
            with open(os.path.join(site, "myduino_labs.pth"), "w") as f:       # the notebooks import labview
                f.write(os.path.join(HERE, "labs-student") + "\n")
            open(os.path.join(venv, ".ready"), "w").close()
            failed.pop(key, None)
            log("Python environment ready:", key.replace("__", "/"))
        except (subprocess.CalledProcessError, OSError) as error:
            shutil.rmtree(venv, ignore_errors=True)
            failed[key] = (time.time(), f"Could not make the Python environment ({error}). See .jupyter/prepare.log in the student folder.")
            log("failed:", key, error)
        finally:
            preparing.pop(key, None)


# ---------------------------------------------------------------- a student's folder and JupyterLab
def migrate_lab_numbers(home):
    """The course got a new Lab 22 (Audio Language Models): the old Labs 22 to 40 are now 23 to 41. A student folder made before that still has the
    old work in lab22 ... lab40. This moves it up by one, once, so that each lab keeps the work of the student. The marker file says it is done."""
    marker = os.path.join(home, ".lab_numbering")
    if os.path.exists(marker):
        return
    old_layout = bool(glob.glob(os.path.join(home, "lab22", "lab22_multimodal*")))
    if old_layout:
        for n in range(MAX_LABS - 2, 21, -1):                      # from the top, so that nothing is overwritten
            src, dst = os.path.join(home, f"lab{n}"), os.path.join(home, f"lab{n + 1}")
            if not os.path.isdir(src):
                continue
            if os.path.isdir(dst) and not os.listdir(dst):
                os.rmdir(dst)
            if os.path.exists(dst):
                continue                                           # not empty: leave it, never overwrite the work of a student
            os.rename(src, dst)
            for name in os.listdir(dst):
                if name.startswith(f"lab{n}_"):
                    os.rename(os.path.join(dst, name), os.path.join(dst, f"lab{n + 1}_" + name[len(f"lab{n}_"):]))
            os.makedirs(src, exist_ok=True)
        book = os.path.join(home, ".delivered.json")               # what the student was given: the file names change too
        try:
            given = json.load(open(book))
        except (OSError, ValueError):
            given = {}
        moved = {}
        for key, value in given.items():
            m = re.match(r"lab(\d+)/lab(\d+)_(.*)$", key)
            if m and int(m.group(1)) >= 22:
                key = f"lab{int(m.group(1)) + 1}/lab{int(m.group(2)) + 1}_{m.group(3)}"
            moved[key] = value
        with open(book, "w") as f:
            json.dump(moved, f)
    with open(marker, "w") as f:
        f.write("2\n")


def prepare(home, labs=None):
    """labs: the numbers of the labs of the package of this teacher (None: every lab of the course)."""
    migrate_lab_numbers(home)
    for n in range(MAX_LABS):
        if labs is not None and n not in labs:
            continue
        os.makedirs(os.path.join(home, f"lab{n}"), exist_ok=True)
    for d in ("models", ".jupyter", ".ipython", ".config", ".tmp"):
        os.makedirs(os.path.join(home, d), exist_ok=True)
    # JupyterLab saves the open notebook every 20 seconds, so the teacher's Progress tab is never far behind (the default is 2 minutes).
    dm = os.path.join(home, ".jupyter", "settings", "@jupyterlab", "docmanager-extension")
    os.makedirs(dm, exist_ok=True)
    if not os.path.exists(os.path.join(dm, "plugin.jupyterlab-settings")):          # a setting the student made is kept
        with open(os.path.join(dm, "plugin.jupyterlab-settings"), "w") as f:
            json.dump({"autosaveInterval": 20}, f)
    # The student's own bash: the Terminal tab of their JupyterLab starts in their folder, with their environment on.
    rc = os.path.join(home, ".bashrc")
    if "# myduino-lab" not in read(rc):
        with open(rc, "a") as f:
            f.write('# myduino-lab\n[ -f "$HOME/.venv/bin/activate" ] && source "$HOME/.venv/bin/activate"\n'
                    'cd "$HOME" 2>/dev/null\n')
    if "alias lab=" in read(rc):                                   # the old lab command is gone
        with open(rc, "w") as f:
            f.write("".join(l for l in read(rc).splitlines(True) if not l.startswith("alias lab=")))
    with open(os.path.join(home, ".bash_profile"), "w") as f:
        f.write('[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"\n')
    for pth in glob.glob(os.path.join(home, ".venv", "lib", "python*", "site-packages", "myduino_labs.pth")):   # keep the path right if the course folder moves
        with open(pth, "w") as f:
            f.write(os.path.join(HERE, "labs-student") + "\n")
    shutil.copyfile(os.path.join(HERE, "labs-student", "start_here.ipynb"), os.path.join(home, "Start_Here.ipynb"))
    sync_notebooks(home, labs)


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sync_notebooks(home, labs=None):
    """Gives the student the notebooks of the course. A new lab is copied. A notebook the student never changed is replaced by the newer
    version of the course. A notebook the student changed is kept as it is, and the newer version is added next to it as "<name> (new).ipynb"."""
    course = os.path.join(HERE, "labs-student")
    book = os.path.join(home, ".delivered.json")                   # what the student was given: file -> {"sent": hash of that version, "new": hash of the newer version put next to it}
    try:
        given = json.load(open(book))
    except (OSError, ValueError):
        given = {}
    for folder in sorted(os.listdir(course)):
        m = re.match(r"lab(\d{1,2})_", folder)
        if not m or not os.path.isdir(os.path.join(course, folder)):
            continue
        if labs is not None and int(m.group(1)) not in labs:
            continue                                                # a lab that is not in the package
        for f in os.listdir(os.path.join(course, folder)):
            if not f.endswith(".ipynb"):
                continue
            src, dest = os.path.join(course, folder, f), os.path.join(home, f"lab{m.group(1)}", f)
            rel = f"lab{m.group(1)}/{f}"
            new, rec = file_hash(src), given.get(rel, {})
            if not os.path.exists(dest):
                shutil.copyfile(src, dest)
                given[rel] = {"sent": new}
            elif file_hash(dest) == new:
                given[rel] = {"sent": new}                          # the student has the current version
            elif rec.get("sent") and file_hash(dest) == rec["sent"]:
                shutil.copyfile(src, dest)                          # never changed by the student: the newer version replaces it
                given[rel] = {"sent": new}
            elif rec.get("new") != new:
                shutil.copyfile(src, dest[:-6] + " (new).ipynb")    # changed by the student: their work stays, the newer version comes next to it
                given[rel] = {**rec, "new": new}
    tmp = book + ".tmp"
    with open(tmp, "w") as f:
        json.dump(given, f)
    os.replace(tmp, book)

def student_env(home, cls, camera_device, pin=None, mqtt_port=None, lab=None):
    env = dict(os.environ)
    venv = venv_of(home)
    env.update(
        # HOME is the student folder, so everything the programs write (settings, caches, history, temporary files)
        # lands in the student's own folder and nothing is left anywhere else on the Jetson.
        LD_LIBRARY_PATH=os.path.join(venv, "lib", "python3.10", "site-packages", "nvidia", "cu12", "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        HOME=home, TMPDIR=os.path.join(home, ".tmp"), PIP_CACHE_DIR=os.path.join(ROOT, ".cache", "pip"), LAB_ROOT=ROOT,
        VIRTUAL_ENV=venv, PATH=os.path.join(venv, "bin") + os.pathsep + env.get("PATH", ""),
        LAB_HOME=home, LAB_CLASS=cls, HISTFILE=os.path.join(home, ".bash_history"),
        JUPYTER_CONFIG_DIR=os.path.join(home, ".jupyter", "config"), JUPYTER_DATA_DIR=os.path.join(home, ".jupyter", "data"),
        JUPYTER_RUNTIME_DIR=os.path.join(home, ".jupyter", "runtime"), JUPYTERLAB_SETTINGS_DIR=os.path.join(home, ".jupyter", "settings"),
        JUPYTERLAB_WORKSPACES_DIR=os.path.join(home, ".jupyter", "workspaces"), IPYTHONDIR=os.path.join(home, ".ipython"),
        YOLO_CONFIG_DIR=os.path.join(home, ".config", "Ultralytics"), MPLCONFIGDIR=os.path.join(home, ".config", "matplotlib"),
    )
    env.pop("PYTHONHOME", None)
    if lab is not None:                                    # each lab keeps its own JupyterLab layout (the open tabs of the lab before do not come back)
        env["JUPYTERLAB_WORKSPACES_DIR"] = os.path.join(home, ".jupyter", "workspaces", f"lab{lab}")
        os.makedirs(env["JUPYTERLAB_WORKSPACES_DIR"], exist_ok=True)
    if pin is not None and mqtt_port:                      # the student's own MQTT broker (see the MQTT section)
        env.update(LAB_MQTT_HOST=MQTT_HOST[0], LAB_MQTT_PORT=str(mqtt_port), LAB_MQTT_USER=os.path.basename(home), LAB_MQTT_PASS=str(pin))
    if camera_device:
        env["LAB_CAMERA"] = camera_device
    else:
        env.pop("LAB_CAMERA", None)
    for key in ("PIP_CACHE_DIR", "TMPDIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR", "JUPYTERLAB_SETTINGS_DIR", "JUPYTERLAB_WORKSPACES_DIR", "YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        os.makedirs(env[key], exist_ok=True)
    return env


def start(cls, student, port, pin, camera, mqtt_port=None, lab=None, labs=None):
    home = os.path.join(ROOT, cls, student)
    prepare(home, labs)
    mqtt_port = mqtt_port or int(port) - 7000
    device = next((c["device"] for c in cameras() if c["number"] == camera), "") if camera is not None else ""
    # The teacher started one lab: JupyterLab shows only that lab's folder and opens its notebook. Without a lab it shows the whole folder (old way).
    root, first = home, "Start_Here.ipynb"
    if lab is not None and os.path.isdir(os.path.join(home, f"lab{lab}")):
        root = os.path.join(home, f"lab{lab}")
        books = sorted(f for f in os.listdir(root) if f.endswith(".ipynb"))
        first = books[0] if books else ""
    log("start", f"{cls}/{student}", "port", port, "lab", lab if lab is not None else "-", "camera", camera if camera is not None else "-")
    out = open(os.path.join(home, ".jupyter", "server.log"), "ab")
    proc = subprocess.Popen(
        [os.path.join(venv_of(home), "bin", "jupyter"), "lab", "--no-browser", "--ip=0.0.0.0", f"--port={port}", "--ServerApp.port_retries=0",
         f"--ServerApp.token={pin}", "--ServerApp.allow_remote_access=True", f"--ServerApp.root_dir={root}"] + ([first] if first else []),
        cwd=root, env=student_env(home, cls, device, pin, mqtt_port, lab), stdout=out, stderr=out, start_new_session=True)
    running[f"{cls}__{student}"] = {"proc": proc, "port": port, "pin": pin, "camera": camera, "lab": lab, "started": time.time(),
                                    "mqtt": mqtt_start(home, student, pin, mqtt_port), "mqtt_port": mqtt_port}


def stop(key):
    r = running.pop(key, None)
    if not r:
        return
    log("stop", key.replace("__", "/"))
    mqtt_stop(r.get("mqtt"))
    try:
        os.killpg(r["proc"].pid, signal.SIGTERM)
        r["proc"].wait(timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(r["proc"].pid, signal.SIGKILL)
        except OSError:
            pass


def delete_folder(cls, student):
    if not (NAME.match(cls) and NAME.match(student)):
        return
    key = f"{cls}__{student}"
    stop(key)
    failed.pop(key, None)
    target = os.path.realpath(os.path.join(ROOT, cls, student))
    if os.path.dirname(os.path.dirname(target)) == os.path.realpath(ROOT) and os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
        log("deleted", f"{cls}/{student}")
        try:
            os.rmdir(os.path.join(ROOT, cls))                       # the class folder goes too, once the last student is gone
        except OSError:
            pass
    deleted.append(key)


def rename_class(old, new):
    """The teacher renamed a class: move the class folder, then fix the paths inside the students' Python environments."""
    if not (NAME.match(old) and NAME.match(new)) or old == new:
        return
    tag = f"{old}>{new}"
    src, dst = os.path.join(ROOT, old), os.path.join(ROOT, new)
    for key in [k for k in running if k.startswith(old + "__")]:
        stop(key)
    try:
        if os.path.isdir(src):
            if os.path.isdir(dst):
                os.rmdir(dst)                                      # an empty folder of an old class of that name is fine; a full one raises
            os.rename(src, dst)
            for student in os.listdir(dst):
                fix_paths(os.path.join(dst, student), os.path.join(src, student), os.path.join(dst, student))
            log("renamed class", old, "to", new)
        rename_failed.pop(tag, None)
        renamed.append(tag)
    except OSError as error:
        if rename_failed.get(tag) != str(error):
            log("cannot rename class", old, "to", new, ":", error)
        rename_failed[tag] = str(error)


def fix_paths(home, old_home, new_home):
    """A Python environment has its own folder path inside its scripts. After a move, write the new path."""
    venv = os.path.join(home, ".venv")
    if not os.path.isdir(venv):
        return
    old_b, new_b = old_home.encode(), new_home.encode()
    for folder, _, files in os.walk(os.path.join(venv, "bin")):
        for name in files:
            path = os.path.join(folder, name)
            try:
                if os.path.islink(path) or os.path.getsize(path) > 2_000_000:
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                if old_b in data and b"\0" not in data[:512]:       # a script, not a program
                    with open(path, "wb") as f:
                        f.write(data.replace(old_b, new_b))
            except OSError:
                pass
    cfg = os.path.join(venv, "pyvenv.cfg")
    try:
        text = open(cfg).read()
        if old_home in text:
            open(cfg, "w").write(text.replace(old_home, new_home))
    except OSError:
        pass


def packages():
    try:
        return json.load(open(os.path.join(HERE, "labs-student", "packages.json")))
    except (OSError, ValueError):
        return []


def solo_labs():
    """The labs of the Solo Lab (a set of numbers), or None when this Jetson has no Solo Lab. It is set with labs-setup.sh --solo <package>."""
    solo = CFG.get("solo")
    if not isinstance(solo, dict):
        return None
    if isinstance(solo.get("labs"), list):
        return {int(n) for n in solo["labs"] if isinstance(n, int) and 0 <= n < MAX_LABS}
    pack = next((p for p in packages() if p.get("id") == solo.get("package")), None)
    return set(range(int(pack["from"]), int(pack["to"]) + 1)) if pack else None


def ensure_solo_pin():
    """The Solo Lab has a PIN of 6 digits that stays the same (it is saved in the settings): the student opens the lab from a laptop with the address and this PIN."""
    solo = CFG.get("solo")
    if not isinstance(solo, dict) or re.fullmatch(r"\d{6}", str(solo.get("pin", ""))):
        return
    solo["pin"] = str(secrets.randbelow(900000) + 100000)
    try:
        cfg = json.load(open(CONFIG))
    except (OSError, ValueError):
        cfg = dict(CFG)
    if isinstance(cfg.get("solo"), dict):
        cfg["solo"]["pin"] = solo["pin"]
    else:
        cfg["solo"] = dict(solo)
    tmp = CONFIG + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(cfg, f)
    os.replace(tmp, CONFIG)


def solo_entry():
    """The plan entry of the Solo Lab: one student, the whole folder of the package, JupyterLab on port 8880 with the PIN of the Solo Lab."""
    labs = solo_labs()
    camera = running[SOLO_KEY]["camera"] if SOLO_KEY in running else (0 if cameras() else None)     # the camera stays as it was while the lab runs
    return {"class": "solo", "student": "student", "port": SOLO_PORT, "pin": str(CFG["solo"].get("pin", "")), "camera": camera, "mqtt_port": SOLO_PORT - 7000,
            "lab": None, "labs": sorted(labs) if labs is not None else None}


def solo_possible():
    """The Solo Lab runs whenever the teacher app has no lab for this Jetson: there is no teacher app, it does not answer, or it answers but has not
    put a student here. The lab of a class on this Jetson comes first (the Solo Lab closes, and starts again when the class lab ends)."""
    return solo_labs() is not None and not last_plan


def solo_active():
    """True when a student used the Solo Lab in the last 5 minutes (a kernel is open and JupyterLab saw activity)."""
    r = running.get(SOLO_KEY)
    if not r or r["proc"].poll() is not None:
        return False
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{SOLO_PORT}/api/status", headers={"Authorization": f"token {r['pin']}"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            st = json.load(resp)
        seen = time.mktime(time.strptime(st["last_activity"][:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone      # the time is in UTC
        return int(st.get("kernels", 0)) > 0 and time.time() - seen < 300
    except (OSError, ValueError, KeyError):
        return False


def solo_report():
    """What the teacher app is told about the Solo Lab of this Jetson (None: it has none)."""
    labs = solo_labs()
    if labs is None:
        return None
    pack = next((p for p in packages() if p.get("id") == CFG["solo"].get("package")), {})
    up = SOLO_KEY in running and running[SOLO_KEY]["proc"].poll() is None
    return {"package": str(pack.get("id", "")), "title": str(pack.get("title", "")), "labs": len(labs), "running": up, "active": up and solo_active(),
            "preparing": SOLO_KEY in preparing, "port": SOLO_PORT, "pin": str(CFG["solo"].get("pin", ""))}


def plan_with_solo(plan):
    """The plan of the teacher app wins. With nothing for this Jetson in it, the Solo Lab is in the plan (it is always ready for a student)."""
    if plan or solo_labs() is None:
        return plan
    return plan + [solo_entry()]


def package_labs(w):
    """The labs the teacher has in the package (a set of numbers), or None when the teacher app does not say (an older app: every lab)."""
    labs = w.get("labs")
    return {int(n) for n in labs if isinstance(n, int) and 0 <= n < MAX_LABS} if isinstance(labs, list) else None


def reconcile(plan):
    wanted = {}
    for p in plan:
        if NAME.match(str(p.get("class", ""))) and NAME.match(str(p.get("student", ""))):
            wanted[f"{p['class']}__{p['student']}"] = p
    for key in list(running):
        r = running[key]
        w = wanted.get(key)
        if w is None or r["proc"].poll() is not None or (r["port"], r["pin"], r["camera"], r.get("lab")) != (w["port"], str(w["pin"]), w.get("camera"), w.get("lab")):
            gone = w is not None and r["proc"].poll() is not None
            stop(key)
            if gone and time.time() - r["started"] < 15:
                wanted[key]["_wait"] = True                                  # it stops at once: do not restart in a loop
    for key, r in running.items():                                           # a broker that stopped by itself starts again
        if r.get("mqtt") is not None and r["mqtt"].poll() is not None:
            home = os.path.join(ROOT, *key.split("__", 1))
            r["mqtt"] = mqtt_start(home, key.split("__", 1)[1], r["pin"], r["mqtt_port"])
    for key, w in wanted.items():
        home = os.path.join(ROOT, w["class"], w["student"])
        if key not in running and not venv_ready(home):            # the student's own Python environment first
            if key not in preparing and time.time() - failed.get(key, (0, ""))[0] > 60:
                preparing[key] = time.time()
                threading.Thread(target=make_venv, args=(key, home), daemon=True).start()
            continue
        if key not in running and not w.get("_wait"):
            try:
                start(w["class"], w["student"], int(w["port"]), str(w["pin"]), w.get("camera"), w.get("mqtt_port"), w.get("lab"), package_labs(w))
            except OSError as error:
                log("cannot start", key, error)


# ---------------------------------------------------------------- MQTT broker of each student (ESP32 boards, phone apps)
MQTT_HOST = [""]                  # the address of this Jetson as the teacher sees it (set by hello)


def mqtt_ready():
    return bool(shutil.which("mosquitto") and shutil.which("mosquitto_passwd"))


def mqtt_start(home, user, pin, port):
    """The student's own MQTT broker, on its own port like their JupyterLab: user = the student's name, password = the PIN.
    It is a program of the agent (no sudo needed) and stops with the student's lab."""
    if not mqtt_ready():
        return None
    folder = os.path.join(home, ".mqtt")
    os.makedirs(folder, exist_ok=True)
    passwd, conf = os.path.join(folder, "passwd"), os.path.join(folder, "mosquitto.conf")
    try:
        subprocess.run(["mosquitto_passwd", "-c", "-b", passwd, user, str(pin)], check=True, capture_output=True)
        with open(conf, "w") as f:
            f.write(f"listener {port} 0.0.0.0\nallow_anonymous false\npassword_file {passwd}\npersistence false\n")
        for f in (passwd, conf):
            os.chmod(f, 0o600)
        return subprocess.Popen(["mosquitto", "-c", conf], stdout=open(os.path.join(folder, "mosquitto.log"), "ab"), stderr=subprocess.STDOUT,
                                start_new_session=True)
    except (subprocess.CalledProcessError, OSError) as error:
        log("cannot start the MQTT broker of", user, error)
        return None


def mqtt_stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------- progress: read from the students' saved notebooks
progress_at, progress_last = 0, {}
sizes = {}            # "class__student" -> megabytes, measured in the background (du on a Python environment takes seconds)


def folder_keys():
    """class__student of every student folder on this Jetson, known to the teacher or not."""
    out = []
    try:
        for c in sorted(os.listdir(ROOT)):
            if NAME.match(c) and c != "solo" and os.path.isdir(os.path.join(ROOT, c)):     # the Solo Lab is not a class of the teacher
                out += [f"{c}__{n}" for n in sorted(os.listdir(os.path.join(ROOT, c))) if NAME.match(n) and os.path.isdir(os.path.join(ROOT, c, n))]
    except OSError:
        pass
    return out


def measure_sizes():
    while True:
        try:
            keys = folder_keys()
            for k in list(sizes):
                if k not in keys:
                    del sizes[k]
            for k in keys:
                cls, stu = k.split("__", 1)
                out = subprocess.run(["du", "-sm", os.path.join(ROOT, cls, stu)], capture_output=True, text=True, timeout=300).stdout.split()
                if out:
                    sizes[k] = int(out[0])
        except Exception:
            pass
        time.sleep(300)

progress_cache = {}     # notebook path -> (mtime, done, total, errors)


def notebook_progress(path):
    try:
        mtime = os.path.getmtime(path)
        hit = progress_cache.get(path)
        if hit and hit[0] == mtime:
            return hit
        done = total = errors = 0
        for c in json.load(open(path)).get("cells", []):
            if c.get("cell_type") != "code" or not "".join(c.get("source", [])).strip():
                continue
            total += 1
            if c.get("execution_count") is not None:
                done += 1
            if any(o.get("output_type") == "error" for o in c.get("outputs", [])):
                errors += 1
        progress_cache[path] = hit = (mtime, done, total, errors)
        return hit
    except (OSError, ValueError):
        return None


live_sent = {}        # key -> modified time of the last live picture sent


def live_report():
    """Small pictures of the students' running camera views (labview writes .tmp/live.jpg about every 1.5 seconds)."""
    out = {}
    for key in list(running)[:8]:
        cls, _, stu = key.partition("__")
        path = os.path.join(ROOT, cls, stu, ".tmp", "live.jpg")
        try:
            mtime = os.stat(path).st_mtime
            if time.time() - mtime > 8:
                live_sent.pop(key, None)
                continue
            if live_sent.get(key) == mtime:
                out[key] = ""                                  # same picture as before: the teacher keeps the one it has
                continue
            with open(path, "rb") as f:
                data = f.read()
            if len(data) < 250000:
                out[key] = base64.b64encode(data).decode()
                live_sent[key] = mtime
        except OSError:
            live_sent.pop(key, None)
    return out


def progress_report():
    out = {}
    try:
        classes = [c for c in os.listdir(ROOT) if NAME.match(c) and c != "solo" and os.path.isdir(os.path.join(ROOT, c))]
    except OSError:
        return out
    for cls in classes:
        for stu in os.listdir(os.path.join(ROOT, cls)):
            home = os.path.join(ROOT, cls, stu)
            if not NAME.match(stu) or not os.path.isdir(home):
                continue
            labs, last = {}, 0
            for n in range(MAX_LABS):
                for f in glob.glob(os.path.join(home, f"lab{n}", "*.ipynb")):
                    p = notebook_progress(f)
                    if p and p[2]:
                        labs[str(n)] = [p[1], p[2], p[3]]
                        if p[1]:
                            last = max(last, p[0])
            out[f"{cls}__{stu}"] = {"labs": labs, "idle": int(time.time() - last) if last else None}
    return out


# ---------------------------------------------------------------- talking to the teacher server
def course_version():
    """The commit of the course folder on this Jetson (the teacher compares it with the newest one on GitHub)."""
    try:
        head = read(os.path.join(HERE, ".git", "HEAD")).strip()
        if head.startswith("ref:"):
            head = read(os.path.join(HERE, ".git", head[4:].strip())).strip()
        return head[:40]
    except Exception:
        return ""


def desktop_on():
    """True when the screen desktop (GDM) of this Jetson runs. The teacher can turn it off to free about 1 GB of memory."""
    try:
        return subprocess.run(["systemctl", "is-active", "--quiet", "gdm"], timeout=3).returncode == 0
    except Exception:
        return None


_cpu_last = [None]


def stats():
    """CPU, GPU, memory, disk and temperature of this Jetson, for the teacher's Jetsons page. Anything unreadable is left out."""
    out = {}
    try:
        f = [int(x) for x in read("/proc/stat").split("\n", 1)[0].split()[1:]]
        total, idle = sum(f), f[3] + f[4]
        last, _cpu_last[0] = _cpu_last[0], (total, idle)
        if last and total > last[0]:
            out["cpu"] = round(100 * (1 - (idle - last[1]) / (total - last[0])))
    except Exception:
        pass
    for p in ("/sys/devices/platform/gpu.0/load", "/sys/devices/platform/bus@0/17000000.gpu/load"):
        try:
            out["gpu"] = round(int(read(p).strip()) / 10)
            break
        except Exception:
            pass
    try:
        m = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in read("/proc/meminfo").splitlines() if ":" in l)}
        out["ram"] = [round((m["MemTotal"] - m["MemAvailable"]) / 1048576, 1), round(m["MemTotal"] / 1048576, 1)]      # GB used, GB total
    except Exception:
        pass
    try:
        d = shutil.disk_usage(os.path.expanduser("~"))
        out["disk"] = [round(d.used / 1e9), round(d.total / 1e9)]
    except Exception:
        pass
    try:
        out["temp"] = round(int(read("/sys/class/thermal/thermal_zone0/temp").strip()) / 1000)
    except Exception:
        pass
    return out


def hello(cfg):
    global progress_at, progress_last
    host, _, _ = cfg["teacher"].partition(":")
    MQTT_HOST[0] = my_ip(host)
    if time.time() - progress_at > 9:                                # notebooks are read every 10 seconds, not every 5
        progress_at, progress_last = time.time(), progress_report()
    body = {
        "id": cfg.get("id") or re.sub(r"[^a-z0-9_]", "_", socket.gethostname().lower()), "ip": MQTT_HOST[0], "cameras": cameras(),
        "running": [{"key": k, "port": r["port"], "alive": r["proc"].poll() is None} for k, r in running.items()],
        "folders": [{"key": k, "mb": sizes.get(k, 0)} for k in folder_keys()], "progress": progress_last, "live": live_report(), "deleted": list(deleted), "renamed": list(renamed), "preparing": list(preparing), "ready": True, "solo": solo_report(), "course": course_version(), "stats": stats(), "desktop": desktop_on(),
        "note": next(iter(m for _, m in failed.values()), "") or ("MQTT is not installed on this Jetson. Click Update on the Jetsons page." if running and not mqtt_ready() else "") or next(iter(f"Could not rename a class folder: {m}" for m in rename_failed.values()), ""),
    }
    req = urllib.request.Request(f"http://{cfg['teacher']}/agent/hello", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=6) as r:
        answer = json.load(r)
    del deleted[:]
    del renamed[:]
    return answer


# ---------------------------------------------------------------- the desktop icon: opens the JupyterLab of the student who works at this Jetson
def launcher_page():
    """(status, headers, body) for the page the desktop icon opens. One student: straight into their JupyterLab."""
    live = [(k, r) for k, r in list(running.items()) if r["proc"].poll() is None]
    if len(live) == 1:
        r = live[0][1]
        return 302, {"Location": f"http://localhost:{r['port']}/lab?token={r['pin']}"}, b""
    if not live and solo_possible():                                         # this Jetson gives its own lab to the student at its screen (it starts by itself)
        if SOLO_KEY in failed:
            text = f"<h1>Your lab could not start</h1><p>{html.escape(failed[SOLO_KEY][1])}</p><p>It tries again by itself. Reload this page in a minute.</p>"
        else:
            text = ("<meta http-equiv=refresh content=5><h1>Getting your lab ready...</h1>"
                    "<p>The first time it takes a few minutes. This page opens your lab by itself.</p>")
    elif not live:
        text = "<h1>No lab yet</h1><p>Your teacher has not started a lab for you. Try again in a minute.</p>"
    else:                                                                    # several students on this Jetson: each one signs in with their own PIN
        links = "".join(f'<li><a href="http://localhost:{r["port"]}/">{html.escape(k.split("__", 1)[-1].replace("_", " ").title())}</a></li>' for k, r in live)
        text = f"<h1>Who are you?</h1><ul>{links}</ul><p>Type your PIN on the next page.</p>"
    page = ("<!doctype html><meta charset=utf-8><title>Myduino Jetson Lab</title><style>body{font:18px sans-serif;max-width:30em;margin:15vh auto;padding:0 1em}"
            "a{font-size:1.2em}</style>" + text).encode()
    return 200, {"Content-Type": "text/html; charset=utf-8"}, page


def info_page():
    """The address and the PIN to open the Solo Lab from a laptop on the same network."""
    if solo_labs() is None:
        text = "<h1>No Solo Lab</h1><p>This Jetson has no Solo Lab. Your teacher starts your lab.</p>"
    else:
        ip = my_ip(str(CFG.get("teacher", "")).partition(":")[0] or "10.255.255.255")
        text = (f"<h1>Open your lab from a laptop</h1><p>On the same network, open this address in a browser:</p><p><b>http://{ip}:{SOLO_PORT}</b></p>"
                f"<p>PIN: <b>{html.escape(str(CFG['solo'].get('pin', '')))}</b></p>")
    page = ("<!doctype html><meta charset=utf-8><title>Myduino Jetson Lab</title><style>body{font:18px sans-serif;max-width:30em;margin:15vh auto;padding:0 1em}"
            "b{font-size:1.3em}</style>" + text).encode()
    return 200, {"Content-Type": "text/html; charset=utf-8"}, page


class Launcher(http.server.BaseHTTPRequestHandler):
    def answer(self, with_body):
        code, headers, body = info_page() if self.path.startswith("/info") else launcher_page()
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def do_GET(self):
        self.answer(True)

    def do_HEAD(self):
        self.answer(False)

    def log_message(self, *a):
        pass


def start_launcher():
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", LAUNCH_PORT), Launcher)   # localhost only: the PINs never leave this Jetson
    except OSError as error:
        log("desktop icon page not started:", error)
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main():
    try:
        cfg = json.load(open(CONFIG))
    except (OSError, ValueError):
        sys.exit(f"No settings in {CONFIG}. Run labs-setup.sh <teacher address> or labs-setup.sh --solo <package>.")
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    CFG.update(cfg)
    ensure_solo_pin()
    log("agent started; teacher", cfg.get("teacher") or "none", "| Solo Lab", "on" if solo_labs() is not None else "off")
    start_launcher()
    threading.Thread(target=measure_sizes, daemon=True).start()
    shown = ""
    last_ok = time.time()
    try:
        while True:
            if not cfg.get("teacher"):                                       # no teacher app at all: only the Solo Lab
                reconcile(plan_with_solo([]))
                time.sleep(EVERY)
                continue
            try:
                answer = hello(cfg)
                last_ok = time.time()
                for r in answer.get("rename", []):
                    rename_class(str(r.get("from", "")), str(r.get("to", "")))
                for d in answer.get("delete", []):
                    delete_folder(str(d.get("class", "")), str(d.get("student", "")))
                last_plan[:] = answer.get("plan", [])
                reconcile(plan_with_solo(last_plan))
                shown = ""
            except (urllib.error.URLError, OSError, ValueError) as error:
                if str(error) != shown:                                      # say it once, not every 5 seconds
                    log("cannot reach the teacher server:", error)
                    shown = str(error)
                if any(k != SOLO_KEY for k in running) and time.time() - last_ok > GRACE:      # the teacher PC is gone: close the labs of the class
                    log("no news from the teacher for", GRACE // 60, "minutes: stopping all student sessions")
                    for key in [k for k in running if k != SOLO_KEY]:
                        stop(key)
                    del last_plan[:]
                if solo_labs() is not None and not last_plan:
                    reconcile(plan_with_solo([]))                            # the Solo Lab goes on (or starts) without the teacher app
            time.sleep(EVERY)
    finally:
        for key in list(running):
            stop(key)


if __name__ == "__main__":
    main()
