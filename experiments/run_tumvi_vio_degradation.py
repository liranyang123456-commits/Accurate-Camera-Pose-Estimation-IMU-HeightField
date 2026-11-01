#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled degradation of official ORB-SLAM3 mono-inertial on TUM-VI room1, one condition at a time: the lever arm of
T_b_c1 set to zero (the primary rig has no calibrated t_CI), a constant offset added to the IMU timestamps (0.02 and
0.05 s, the residual of an offline latency estimate; 0.50 s, the uncompensated latency of the primary rig), and the frame
rate lowered to 6.7 Hz with the zero lever arm, alone and with the 0.02 s offset. Every other setting is that of the
frame-rate sweep (run_tumvi_framerate.py). Trajectories are scored like the sweep (ATE after SE(3) alignment, 0.5 s RPE)
and kept for the window-level comparison of eval_tumvi_fusion.py. Writes tumvi_vio_degradation_results.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_tumvi_framerate as rtf  # noqa: E402


def wsl(cmd, timeout=3600):
    """Run in WSL; stdout only (WSL may print a localhost-proxy notice on stderr)."""
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout[-3000:]


def prepare(strides):
    ts = sorted(p.stem for p in (rtf.TUM / "extracted" / "dataset-room1_512" / "mav0" / "cam0" / "data").glob("*.png"))
    for k in strides:
        (rtf.FR / f"ts_stride{k}.txt").write_text("\n".join(ts[::k]) + "\n", encoding="utf-8", newline="\n")
        rc, log = wsl(f"cp {rtf.WSL_TUM}/framerate/ts_stride{k}.txt /root/orbslam3_ws/ts_stride{k}.txt && wc -l < /root/orbslam3_ws/ts_stride{k}.txt")
        assert rc == 0 and int(log.strip().splitlines()[-1]) == len(ts[::k]), log[-300:]

OUT = HERE / "tumvi_vio_degradation_results.json"
DEG = rtf.FR.parent / "degradation"
WSL_DEG = rtf.WSL_TUM + "/degradation"
IMU_SRC = f"{rtf.ORB}/Examples/Monocular-Inertial/TUM_IMU/dataset-room1_512.txt"
LEVER = ("0.045574835649698026", "-0.071161801837997044", "-0.044681254117144367")
VARIANTS = {  # name: (stride, zero lever arm, IMU timestamp offset in s)
    "lever0_20hz": (1, True, 0.0),
    "offset0.02_20hz": (1, False, 0.02),
    "offset0.05_20hz": (1, False, 0.05),
    "offset0.50_20hz": (1, False, 0.50),
    "lever0_6.7hz": (3, True, 0.0),
    "lever0_offset0.02_6.7hz": (3, True, 0.02),
}
SHIFT_PY = """import sys
src, dst, off = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as fin, open(dst, 'w') as fout:
    for ln in fin:
        if ln.startswith('#') or not ln.strip():
            fout.write(ln)
            continue
        a, b = ln.split(',', 1)
        fout.write(str(int(a) + off) + ',' + b)
"""
SHIFT = "python3 /root/orbslam3_ws/shift_imu.py {src} {dst} {off}"


def main():
    names = sys.argv[1:] or list(VARIANTS)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"protocol": __doc__.strip().splitlines()[0], "runs": {}}
    prepare(sorted({VARIANTS[n][0] for n in names}))
    DEG.mkdir(parents=True, exist_ok=True)
    (DEG / "shift_imu.py").write_text(SHIFT_PY, encoding="utf-8", newline="\n")
    rc, _ = wsl(f"cp {WSL_DEG}/shift_imu.py /root/orbslam3_ws/shift_imu.py")
    assert rc == 0
    for name in names:
        k, lever0, off = VARIANTS[name]
        tag = "deg_" + name.replace(".", "p")
        yaml = f"/root/orbslam3_ws/{tag}.yaml"
        sed = f"s/Camera.fps: 20/Camera.fps: {rtf.FPS[k]}/"
        if lever0:
            sed += "".join(f"; s/{v}/0.0/" for v in LEVER)
        imu = f"/root/orbslam3_ws/{tag}_imu.txt"
        prep = (f"sed '{sed}' {rtf.ORB}/Examples/Monocular-Inertial/TUM-VI.yaml > {yaml} && "
                + SHIFT.format(src=IMU_SRC, dst=imu, off=int(round(off * 1e9))) + f" && head -2 {imu} | tail -1 && grep -A 6 T_b_c1 {yaml} | tail -3")
        rc, log = wsl(prep)
        assert rc == 0, log[-500:]
        cmd = (f"cd {rtf.ORB} && rm -f f_{tag}.txt kf_{tag}.txt && timeout 2400 xvfb-run -a ./Examples/Monocular-Inertial/mono_inertial_tum_vi "
               f"Vocabulary/ORBvoc.txt {yaml} {rtf.WSL_TUM}/extracted/dataset-room1_512/mav0/cam0/data /root/orbslam3_ws/ts_stride{k}.txt {imu} {tag} "
               f"> {WSL_DEG}/{tag}.log 2>&1; mkdir -p {WSL_DEG}/{tag}; mv -f f_{tag}.txt kf_{tag}.txt {WSL_DEG}/{tag}/ 2>/dev/null; "
               f"grep -c 'Creation of new map' {WSL_DEG}/{tag}.log; grep -c 'end VIBA 2' {WSL_DEG}/{tag}.log")
        rc, log = wsl(cmd)
        m = rtf.evaluate(DEG / tag / f"f_{tag}.txt", tag)
        maps = [ln for ln in log.strip().splitlines() if ln.strip().isdigit()]
        res["runs"][name] = {"stride": k, "frame_rate_hz": 20.0 / k, "lever_arm_zero": lever0, "imu_offset_s": off,
                             "maps_created": int(maps[0]) if maps else None, "viba2_completed": int(maps[1]) if len(maps) > 1 else None,
                             "trajectory": str(DEG / tag / f"f_{tag}.txt"), **m}
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, res["runs"][name], flush=True)


if __name__ == "__main__":
    main()
