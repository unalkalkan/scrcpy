#!/usr/bin/env python3
"""End-to-end video latency of scrcpy setups: device capture -> frame drawn.

Every frame carries its capture date: the encoder PTS, in the device
CLOCK_MONOTONIC. With SCRCPY_LATENCY_LOG set, scrcpy logs for each frame drawn
its PTS, the date it was received and the date the GPU finished drawing it
(computer CLOCK_MONOTONIC). The device clock is mapped onto the computer clock
by timing round trips to a tiny probe run on the device (adb shell); the error
is at most half of the best round trip, printed with the results.

Measured: device capture (composition of the mirrored display) -> encode ->
USB -> decode -> upscale -> drawn. Not measured: the desktop compositor and the
monitor (the same for every setup), nor input.

    latency/measure.py [-s SERIAL] [-t SECONDS] [--window WxH] [SETUP...] [-- SCRCPY_ARGS...]

SETUP is one of the zsh aliases (scrcpy-game, scrcpy-game-nis, ...) or its
short name (game, nis, vsr, hq, nis-hq, vsr-hq); default: all of them. They run
with the build in build-nis (pass --build), without -G (no virtual gamepad).
Arguments after -- are added to every setup.

Close any running scrcpy first. Keep something animated on the phone (the
game): the device only sends frames when its screen changes.
"""

import argparse
import glob
import hashlib
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "latency"
PROBE_SRC = HERE / "clockprobe.c"
DEVICE_PROBE = "/data/local/tmp/scrcpy-clockprobe"

# The flags of the zsh alias scrcpy-game, without -s and -G
BASE_ARGS = [
    "--no-mouse-hover", "--no-audio", "-m", "1600", "-b", "16M", "-S", "-w",
    "--video-codec-options=vendor.qti-ext-enc-low-latency.enable=1",
]

SETUPS = {
    "game": [],
    "nis": ["--upscale=nis"],
    "vsr": ["--upscale=vsr"],
    "hq": ["-b", "64M"],
    "nis-hq": ["--upscale=nis", "-b", "64M"],
    "vsr-hq": ["--upscale=vsr", "-b", "64M", "--upscale-vsr-quality=17"],
}

NDK_TRIPLES = {
    "arm64-v8a": "aarch64-linux-android",
    "x86_64": "x86_64-linux-android",
}


def die(msg):
    print(f"measure.py: {msg}", file=sys.stderr)
    sys.exit(1)


def proc_cmdlines():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = f.read().split(b"\0")
        except OSError:
            continue
        if args and args[0]:
            yield int(pid), [a.decode(errors="replace") for a in args if a]


def find_adb():
    """The binary of the running adb server: an adb client from another
    release would restart the server (and drop every adb session)."""
    for pid, args in proc_cmdlines():
        if os.path.basename(args[0]) == "adb" and "fork-server" in args:
            try:
                return os.readlink(f"/proc/{pid}/exe")
            except OSError:
                pass
    return os.environ.get("ADB") or "adb"


def adb_cmd(adb, serial, *args, **kwargs):
    return subprocess.run([adb, "-s", serial, *args], capture_output=True,
                          text=True, **kwargs)


def pick_serial(adb, serial):
    if serial:
        return serial
    if os.environ.get("ANDROID_SERIAL"):
        return os.environ["ANDROID_SERIAL"]
    out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
    devices = [line.split()[0] for line in out.splitlines()[1:]
               if line.strip().endswith("device")]
    if len(devices) != 1:
        die(f"{len(devices)} devices attached, pick one with -s")
    return devices[0]


def check_no_scrcpy():
    running = [(pid, args) for pid, args in proc_cmdlines()
               if os.path.basename(args[0]) == "scrcpy"]
    if running:
        lines = "\n".join(f"  {pid}: {shlex.join(args)}" for pid, args in running)
        die("scrcpy is running; close it first (a second mirror shares the "
            f"phone's encoder and the USB link):\n{lines}")


def find_ndk_clang(triple):
    roots = [os.environ.get("ANDROID_NDK_HOME"), os.environ.get("ANDROID_NDK")]
    roots += sorted(glob.glob(str(Path.home() / "Workspace/keymapper/.sdk/ndk/*")))
    roots += sorted(glob.glob(str(Path.home() / "Android/Sdk/ndk/*")))
    for root in filter(None, roots):
        bindir = Path(root) / "toolchains/llvm/prebuilt/linux-x86_64/bin"
        candidates = sorted(bindir.glob(f"{triple}[0-9]*-clang"),
                            key=lambda p: int(p.name[len(triple):-len("-clang")]))
        if candidates:
            return candidates[0]
    die("Android NDK not found (set ANDROID_NDK_HOME)")


def prepare_probe(adb, serial):
    abi = adb_cmd(adb, serial, "shell", "getprop", "ro.product.cpu.abi").stdout.strip()
    triple = NDK_TRIPLES.get(abi)
    if not triple:
        die(f"unsupported device ABI: {abi}")
    binary = HERE / f"clockprobe-{abi}"
    stamp = hashlib.sha256(PROBE_SRC.read_bytes()).hexdigest()[:16]
    stamp_file = binary.with_suffix(".src-sha256")
    if not binary.exists() or not stamp_file.exists() or stamp_file.read_text() != stamp:
        clang = find_ndk_clang(triple)
        subprocess.run([str(clang), "-O2", "-o", str(binary), str(PROBE_SRC)],
                       check=True)
        stamp_file.write_text(stamp)
    r = adb_cmd(adb, serial, "push", str(binary), DEVICE_PROBE)
    if r.returncode:
        die(f"could not push the clock probe: {r.stderr.strip()}")
    adb_cmd(adb, serial, "shell", "chmod", "755", DEVICE_PROBE)


def clock_offset(adb, serial, rounds=300):
    """(device - computer) CLOCK_MONOTONIC offset in ns, and the round trip
    of the sample it comes from (the error is at most half of it)."""
    p = subprocess.Popen([adb, "-s", serial, "shell", "-T", DEVICE_PROBE],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    best = None
    try:
        for _ in range(rounds):
            t0 = time.monotonic_ns()
            p.stdin.write(b"x")
            line = p.stdout.readline()
            t1 = time.monotonic_ns()
            if not line:
                die("the clock probe stopped")
            rtt = t1 - t0
            if best is None or rtt < best[1]:
                best = (int(line) - (t0 + t1) // 2, rtt)
    finally:
        p.stdin.close()
        p.wait(timeout=10)
    return best


def percentile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    i = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[i]


def run_setup(name, args, adb, serial, build, window, duration, warmup, run_dir):
    log = run_dir / f"{name}.frames"
    out = run_dir / f"{name}.scrcpy.log"
    w, h = window
    cmd = [str(build / "app" / "scrcpy"), "-s", serial, *args,
           f"--window-width={w}", f"--window-height={h}",
           f"--window-title=scrcpy latency: {name}",
           f"--time-limit={warmup + duration}"]
    env = dict(os.environ)
    env.setdefault("SDL_VIDEO_DRIVER", "wayland")
    env["ADB"] = adb
    env["SCRCPY_SERVER_PATH"] = str(build / "server" / "scrcpy-server")
    env["SCRCPY_ICON_DIR"] = str(ROOT / "app" / "data")
    env["SCRCPY_LATENCY_LOG"] = str(log)
    vfx = Path(os.environ.get("VFX_LIBS",
                              Path.home() / "Workspace/nvidia-vfx/wheel/nvvfx/libs"))
    if vfx.is_dir():
        env["LD_LIBRARY_PATH"] = f"{vfx}:{env['LD_LIBRARY_PATH']}" \
            if env.get("LD_LIBRARY_PATH") else str(vfx)

    print(f"\n== {name}: {shlex.join(cmd[1:])}", flush=True)
    offset_before, rtt_before = clock_offset(adb, serial)
    with open(out, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=warmup + duration + 60)
    offset_after, rtt_after = clock_offset(adb, serial)
    notes = [line.strip() for line in out.read_text(errors="replace").splitlines()
             if line.startswith(("WARN", "ERROR")) or "Upscaling:" in line
             or "VSR" in line]
    for line in notes:
        print(f"   {line}")
    if r.returncode:
        print(f"   scrcpy exited with {r.returncode}, see {out}")

    stats = analyze(log, (offset_before + offset_after) // 2, warmup)
    stats.update(name=name, clock_error=max(rtt_before, rtt_after) / 2e6,
                 drift=(offset_after - offset_before) / 1e6)
    return stats


def analyze(log, offset, warmup):
    """Latency statistics (ms) of a SCRCPY_LATENCY_LOG file, given the
    (device - computer) clock offset in ns, ignoring the first warmup
    seconds."""
    rows = []
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith("#"):
                continue
            pts_us, recv_us, drawn_us = map(int, line.split())
            if pts_us < 0:
                continue  # no capture date
            capture_ns = pts_us * 1000 - offset  # in the computer clock
            rows.append((capture_ns, None if recv_us < 0 else recv_us * 1000,
                         drawn_us * 1000))
    if rows:
        first = rows[0][2]
        rows = [row for row in rows if row[2] - first >= warmup * 1e9]
    if not rows:
        return {"frames": 0}

    total = sorted((d - c) / 1e6 for c, _, d in rows)
    transit = sorted((r - c) / 1e6 for c, r, _ in rows if r is not None)
    local = sorted((d - r) / 1e6 for _, r, d in rows if r is not None)
    span = (rows[-1][2] - rows[0][2]) / 1e9
    return {
        "frames": len(rows),
        "fps": (len(rows) - 1) / span if span else 0,
        "median": statistics.median(total),
        "p90": percentile(total, 0.9),
        "p99": percentile(total, 0.99),
        "min": total[0],
        "transit": statistics.median(transit) if transit else float("nan"),
        "local": statistics.median(local) if local else float("nan"),
    }


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setups", nargs="*", metavar="SETUP")
    parser.add_argument("-s", "--serial")
    parser.add_argument("-t", "--duration", type=int, default=20,
                        help="seconds measured per setup (default 20)")
    parser.add_argument("--warmup", type=int, default=4,
                        help="seconds ignored at the start of each run (default 4)")
    parser.add_argument("--window", default="2560x1152",
                        help="window size; upscaling cost depends on it (default 2560x1152)")
    parser.add_argument("--build", default=str(ROOT / "build-nis"),
                        help="scrcpy build directory (default build-nis)")
    opts = parser.parse_args(argv)

    names = []
    for s in opts.setups or SETUPS:
        short = s.removeprefix("scrcpy-game").lstrip("-") or "game"
        if short not in SETUPS:
            die(f"unknown setup {s} (one of {', '.join(SETUPS)})")
        names.append(short)
    try:
        window = tuple(int(v) for v in opts.window.lower().split("x"))
        assert len(window) == 2
    except (ValueError, AssertionError):
        die(f"bad --window: {opts.window}")
    build = Path(opts.build)
    if not (build / "app" / "scrcpy").exists():
        die(f"no scrcpy in {build}/app (build it with: ninja -C {build})")

    check_no_scrcpy()
    adb = find_adb()
    serial = pick_serial(adb, opts.serial)
    run_dir = HERE / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    prepare_probe(adb, serial)
    model = adb_cmd(adb, serial, "shell", "getprop", "ro.product.model").stdout.strip()
    print(f"device {serial} ({model}), adb {adb}, window {window[0]}x{window[1]}, "
          f"{opts.duration} s per setup after {opts.warmup} s of warmup")

    results = []
    for name in names:
        results.append(run_setup(name, BASE_ARGS + SETUPS[name] + extra, adb,
                                 serial, build, window, opts.duration,
                                 opts.warmup, run_dir))
        time.sleep(2)

    header = (f"{'setup':<20} {'frames':>6} {'fps':>5} {'median':>7} {'p90':>6} "
              f"{'p99':>6} {'min':>6} {'phone+USB':>9} {'PC':>6}")
    lines = [header, "-" * len(header)]
    for r in results:
        alias = "scrcpy-game" + ("" if r["name"] == "game" else "-" + r["name"])
        if not r["frames"]:
            lines.append(f"{alias:<20} no frames (see {run_dir})")
            continue
        lines.append(
            f"{alias:<20} {r['frames']:>6} {r['fps']:>5.0f} {r['median']:>7.1f} "
            f"{r['p90']:>6.1f} {r['p99']:>6.1f} {r['min']:>6.1f} "
            f"{r['transit']:>9.1f} {r['local']:>6.1f}")
    errors = [r["clock_error"] for r in results if r["frames"]]
    drifts = [abs(r["drift"]) for r in results if r["frames"]]
    if errors:
        lines.append(f"\nms, capture -> drawn. phone+USB: capture -> received; PC: "
                     f"received -> drawn (decode, upscale, draw).\nClock error "
                     f"at most {max(errors):.2f} ms, drift during a run at most "
                     f"{max(drifts):.2f} ms.")
    low = [r["name"] for r in results if r["frames"] and r["fps"] < 30]
    if low:
        lines.append("Few frames per second in: " + ", ".join(low) + ". The phone "
                     "sends frames only when its screen changes; run the game.")
    summary = "\n".join(lines)
    print("\n" + summary)
    (run_dir / "summary.txt").write_text(
        f"{datetime.now():%Y-%m-%d %H:%M}, device {serial} ({model}), window "
        f"{window[0]}x{window[1]}, extra args: {shlex.join(extra) or 'none'}\n\n"
        + summary + "\n")
    print(f"\nRaw logs: {run_dir}")


if __name__ == "__main__":
    main()
