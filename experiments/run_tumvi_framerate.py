#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame-rate sweep of tightly coupled VIO on TUM-VI room1 (hardware-synchronized, Kalibr-calibrated, 20 Hz camera).

Every k-th camera frame (k = 1, 2, 3: 20, 10, 6.7 Hz) is fed to official ORB-SLAM3 mono-inertial and VINS-Fusion
stereo-inertial in WSL, with the full 200 Hz IMU stream and the original calibration. This separates the frame rate
from calibration and synchronization as a cause of initialization failure. Trajectories are scored against the mocap
reference with tumvi_room1/eval_vio_tumvi.py (ATE after SE(3) alignment, 0.5 s RPE). Writes tumvi_framerate_results.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
TUM = IMUCAM_ROOT / "tumvi_room1"
FR = TUM / "framerate"
WSL_TUM = "/mnt/d/reloc3r/tumvi_room1"
ORB = "/root/orbslam3_ws/ORB_SLAM3"
GT = TUM / "extracted" / "dataset-room1_512" / "mav0" / "state_groundtruth_estimate0" / "data.csv"
OUT = HERE / "tumvi_framerate_results.json"
FPS = {1: 20, 2: 10, 3: 7}


def wsl(cmd, timeout=3600):
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def prepare(strides):
    FR.mkdir(exist_ok=True)
    ts = sorted(p.stem for p in (TUM / "extracted" / "dataset-room1_512" / "mav0" / "cam0" / "data").glob("*.png"))
    # Configuration and timestamp files are created inside the WSL file system: files rewritten from Windows on /mnt/d
    # can be read with a stale size by WSL, which made OpenCV's YAML parser fail at the end of the file.
    for k in strides:
        (FR / f"ts_stride{k}.txt").write_text("\n".join(ts[::k]) + "\n", encoding="utf-8", newline="\n")
        rc, log = wsl(f"sed 's/Camera.fps: 20/Camera.fps: {FPS[k]}/' {ORB}/Examples/Monocular-Inertial/TUM-VI.yaml > /root/orbslam3_ws/tumvi_fps{FPS[k]}.yaml && "
                      f"cp {WSL_TUM}/framerate/ts_stride{k}.txt /root/orbslam3_ws/ts_stride{k}.txt && grep -c 'Camera.fps: {FPS[k]}' /root/orbslam3_ws/tumvi_fps{FPS[k]}.yaml")
        assert rc == 0 and log.strip().endswith("1"), log[-300:]
    return {k: len(ts[::k]) for k in strides}


def run_orb(k):
    name = f"tumvi_room1_s{k}"
    cmd = (f"cd {ORB} && rm -f f_{name}.txt kf_{name}.txt && timeout 2400 xvfb-run -a ./Examples/Monocular-Inertial/mono_inertial_tum_vi "
           f"Vocabulary/ORBvoc.txt /root/orbslam3_ws/tumvi_fps{FPS[k]}.yaml {WSL_TUM}/extracted/dataset-room1_512/mav0/cam0/data "
           f"/root/orbslam3_ws/ts_stride{k}.txt Examples/Monocular-Inertial/TUM_IMU/dataset-room1_512.txt {name} > {WSL_TUM}/framerate/orb_s{k}.log 2>&1; "
           f"mkdir -p {WSL_TUM}/framerate/orb_s{k}; mv -f f_{name}.txt kf_{name}.txt {WSL_TUM}/framerate/orb_s{k}/ 2>/dev/null; "
           f"grep -c 'Creation of new map' {WSL_TUM}/framerate/orb_s{k}.log; grep -i 'VIBA\\|Fail\\|reset\\|lost' {WSL_TUM}/framerate/orb_s{k}.log | sort | uniq -c | head")
    rc, log = wsl(cmd)
    return {"wsl_rc": rc, "log_tail": log[-1200:]}


def run_vins(k):
    out = f"{WSL_TUM}/framerate/vins_s{k}"
    cmd = (f"VINS_CFG=/root/vins_ws/our_configs/tumvi_room1_stereo_imu_pinhole_config.yaml bash {WSL_TUM}/run_vins_tumvi_room1_stride.sh "
           f"{WSL_TUM}/extracted_pinhole/dataset-room1_512 {out} 0.5 {k}")
    rc, log = wsl(cmd)
    return {"wsl_rc": rc, "log_tail": log[-800:]}


def evaluate(est: Path, tag: str):
    if not est.exists() or est.stat().st_size == 0:
        return {"num_est_poses": 0, "note": "no trajectory"}
    outj = est.parent / f"metrics_{tag}.json"
    subprocess.run([sys.executable, str(TUM / "eval_vio_tumvi.py"), "--est", str(est), "--gt", str(GT), "--max_dt", "0.02", "--rpe_delta", "0.5",
                    "--out", str(outj)], capture_output=True, text=True)
    if not outj.exists():
        return {"num_est_poses": int(sum(1 for ln in est.read_text().splitlines() if ln and not ln.startswith("#"))), "note": "eval failed"}
    m = json.loads(outj.read_text(encoding="utf-8"))
    return {"num_est_poses": m["num_est_poses"], "num_matched": m["num_matched"], "ate_se3_rmse_m": m["ate_se3_m"]["rmse"],
            "ate_sim3_rmse_m": m["ate_sim3_m"]["rmse"], "rpe_rot_mean_deg": m["rpe_rot_deg"]["mean"], "rpe_trans_mean_m": m["rpe_trans_m"]["mean"],
            "time_span_s": m["time_span_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strides", type=str, default="2,3")
    ap.add_argument("--systems", type=str, default="orb,vins")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="do not re-run a system whose trajectory file exists")
    args = ap.parse_args()
    strides = [int(s) for s in args.strides.split(",")]
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"protocol": __doc__.strip().splitlines()[0], "runs": {}}
    n_frames = prepare(strides)
    for k in strides:
        key = f"stride{k}"
        res["runs"].setdefault(key, {"frame_rate_hz": 20.0 / k, "n_frames": n_frames[k]})
        if "orb" in args.systems:
            if not args.eval_only and not (args.skip_existing and (FR / f"orb_s{k}" / f"f_tumvi_room1_s{k}.txt").exists()):
                res["runs"][key]["orb_run"] = run_orb(k)
            res["runs"][key]["orbslam3_mono_inertial"] = evaluate(FR / f"orb_s{k}" / f"f_tumvi_room1_s{k}.txt", f"orb_s{k}")
        if "vins" in args.systems:
            if not args.eval_only and not (args.skip_existing and (FR / f"vins_s{k}" / "traj_tum.txt").exists()):
                res["runs"][key]["vins_run"] = run_vins(k)
            res["runs"][key]["vins_fusion_stereo_inertial"] = evaluate(FR / f"vins_s{k}" / "traj_tum.txt", f"vins_s{k}")
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(key, {s: res["runs"][key].get(s) for s in ("orbslam3_mono_inertial", "vins_fusion_stereo_inertial")}, flush=True)


if __name__ == "__main__":
    main()
