#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gated fusion with recorded IMU data in the target regime: the nine curated MITI sequences (test split of
build_miti_windows.py; windows of 0.5 s ending at every 20 Hz IR sample; reference from the infrared optical tracker).

Inertial increment: gyroscope rotation (IMU1, calibrated extrinsics, offset estimated on the training segments) and the
translation of the MITI-trained GRU6D (train_miti_gru6d.py, seed 0). Visual front ends at 5.9 Hz (run_visual_miti.py):
height field, ORB-VO, ORB-HVO. Blend weights: the values selected on the checkerboard validation sessions of the primary
rig, applied unchanged ((0.05, 0.40) height field, (0.05, 0.55) ORB-VO, (0.05, 0.85) ORB-HVO), the default gamma = 0.5,
and the ungated gamma_R = 0 variant; no weight is selected on MITI. Writes miti_fusion_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from eval_vio_trajectory_rpe_ate import PoseSeq, _interp_pose, _load_traj_tum, _rel_end_frame  # noqa: E402

VIS = IMUCAM_ROOT / "miti_visual"
PRED = HERE / "miti_gru6d"
OUT = HERE / "miti_fusion_results.json"
TAU = 3.0
W_RIG = {"hf": (0.05, 0.40), "vo": (0.05, 0.55), "hvo": (0.05, 0.85)}
DIRS = {"hf": "hf", "vo": "orb_vo", "hvo": "orb_hvo"}
NONOVL = 10  # every 10th window (0.5 s apart) for the non-overlapping statistics


def geo(A, B):
    return np.degrees(Rotation.from_matrix(np.transpose(A, (0, 2, 1)) @ B).magnitude())


def slerp1(Ra, Rb, w):
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([w]).as_matrix()[0]


def summ(r, t):
    r, t = np.asarray(r, float), np.asarray(t, float)
    return {"rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
            "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)), "N": int(r.size)}


def main():
    z = np.load(HERE / "miti_windows.npz", allow_pickle=True)
    te = z["split"].astype(str) == "test"
    seq = z["test_sequence"].astype(str)[te]
    t0, t1 = z["t_start"][te], z["t_end"][te]
    y = z["y"][te].astype(np.float64)
    R_gt, t_gt = y[:, :9].reshape(-1, 3, 3), y[:, 9:12]
    R_gyro = z["R_gyro"][te]
    names = sorted(set(seq))
    N = len(seq)
    # GRU6D-MITI predictions, per sequence in window order
    res = json.loads((PRED / "miti_gru6d_results.json").read_text(encoding="utf-8"))
    p = np.load(PRED / "pred_gru6d_s0.npz")
    R_gru = np.empty((N, 3, 3))
    t_gru = np.empty((N, 3))
    for n in names:
        m = seq == n
        pr = p[n]
        assert len(pr) == m.sum(), (n, len(pr), m.sum())
        U, _, Vt = np.linalg.svd(pr[:, :9].reshape(-1, 3, 3))
        d = np.sign(np.linalg.det(U @ Vt))
        U[:, :, -1] *= d[:, None]
        R_gru[m], t_gru[m] = U @ Vt, pr[:, 9:12]
    if "shrinkage_gru6d_s0" not in res:  # validation predictions of seed 0 (scalers refitted on the training split, as in training)
        import torch
        from sklearn.preprocessing import StandardScaler
        sys.path.insert(0, str(HERE))
        from train_miti_gru6d import build  # noqa: E402
        import train_inertial_baselines as tib  # noqa: E402
        Xa, Ya, spl = z["X"].astype(np.float32), z["y"].astype(np.float32), z["split"].astype(str)
        itr, iva = np.where(spl == "train")[0], np.where(spl == "val")[0]
        sx = StandardScaler().fit(Xa[itr].reshape(-1, Xa.shape[-1]))
        sp = StandardScaler().fit(Ya[itr])
        model = build("gru6d")
        model.load_state_dict(torch.load(PRED / "gru6d_s0.pt", map_location="cpu"))
        Xv = sx.transform(Xa[iva].reshape(-1, Xa.shape[-1])).reshape(len(iva), Xa.shape[1], Xa.shape[2]).astype(np.float32)
        pv = sp.inverse_transform(tib.predict(model, Xv, "cpu").astype(np.float64))[:, 9:12]
        yv = Ya[iva][:, 9:12]
        curve = {float(s): float(np.linalg.norm(s * pv - yv, axis=1).mean()) for s in np.round(np.arange(0.0, 1.0001, 0.01), 2)}
        res["shrinkage_gru6d_s0"] = {"s_validation": min(curve, key=curve.get), "val_trans_s1": curve[1.0], "val_trans_zero": float(np.linalg.norm(yv, axis=1).mean())}
        print("shrinkage from validation:", res["shrinkage_gru6d_s0"])
    s_val = res["shrinkage_gru6d_s0"]["s_validation"]
    # other retrained encoders and seeds (translation only), in window order per sequence
    enc_t = {}
    for key in sorted(res["runs"]):
        pk = np.load(PRED / f"pred_{key}.npz")
        tt = np.empty((N, 3))
        for n in names:
            tt[seq == n] = pk[n][:, 9:12]
        enc_t[key] = tt
    t_tlio = enc_t["tlio_resnet_s0"]
    # visual increments
    vis = {}
    for key, d in DIRS.items():
        ok = np.zeros(N, bool)
        Rv = np.repeat(np.eye(3)[None], N, 0)
        tv = np.zeros((N, 3))
        for n in names:
            path = VIS / d / n / "traj_tum.txt"
            if not path.exists():
                continue
            est = _load_traj_tum(str(path), time_unit="s", pos_scale=1.0, pose_mode="T_WC")
            for i in np.where(seq == n)[0]:
                e, s = _interp_pose(est, t1[i], max_gap_s=0.2), _interp_pose(est, t0[i], max_gap_s=0.2)
                if e is not None and s is not None:
                    R, tt = _rel_end_frame(e[0], e[1], s[0], s[1])
                    ok[i], Rv[i], tv[i] = True, R, tt.reshape(3)
        vis[key] = (ok, Rv, tv)

    def scales(key, src, ref=False):
        ok, _, tv = vis[key]
        out = {}
        for n in names:
            m = (seq == n) & ok & (np.linalg.norm(tv, axis=1) > 1e-9)
            num = t_gt[m] if ref else src[m]
            out[n] = float(np.median(np.linalg.norm(num, axis=1) / np.linalg.norm(tv[m], axis=1))) if m.any() else 1.0
        return out

    def blend(key, gR, gt, gate=True, tsrc=None, ref_scale=False):
        tsrc = t_gru if tsrc is None else tsrc
        ok, Rv, tv = vis[key]
        sc = scales(key, tsrc, ref_scale)
        R = R_gyro.copy()
        t = tsrc.copy()
        used = 0
        for i in range(N):
            if ok[i] and (not gate or geo(Rv[i][None], R_gyro[i][None])[0] <= TAU):
                used += 1
                if gR > 0:
                    R[i] = slerp1(R_gyro[i], Rv[i], gR)
                t[i] = gt * tv[i] * sc[seq[i]] + (1 - gt) * tsrc[i]
        return geo(R, R_gt), np.linalg.norm(t - t_gt, axis=1), used

    out = {"protocol": __doc__.strip().splitlines()[0], "N": N, "N_non_overlapping": int(len(range(0, N, NONOVL))), "sequences": names,
           "tau": TAU, "shrink_s_validation": s_val, "rows": {}, "rows_non_overlapping": {}, "per_sequence": {}, "coverage": {}, "damage": {}}
    e_gyro = geo(R_gyro, R_gt)
    t_zero = np.linalg.norm(t_gt, axis=1)
    rows = {"zero_motion": (geo(np.repeat(np.eye(3)[None], N, 0), R_gt), t_zero),
            "gyro_zeroT": (e_gyro, t_zero),
            "gru6d_full": (geo(R_gru, R_gt), np.linalg.norm(t_gru - t_gt, axis=1)),
            "inertial": (e_gyro, np.linalg.norm(t_gru - t_gt, axis=1)),
            "inertial_shrunk": (e_gyro, np.linalg.norm(s_val * t_gru - t_gt, axis=1)),
            "inertial_tlio": (e_gyro, np.linalg.norm(t_tlio - t_gt, axis=1))}
    out["encoders_test_trans_mm"] = {k: float(np.linalg.norm(v - t_gt, axis=1).mean()) for k, v in enc_t.items()}
    out["encoders_test_rot_deg"] = {k: res["runs"][k]["test"]["rot_mean"] for k in res["runs"]}
    for key in DIRS:
        ok, Rv, tv = vis[key]
        if not ok.any():
            continue
        sc_ref = scales(key, None, True)
        out["coverage"][key] = {"supported": float(ok.mean()), "gate_accept": float(np.mean([ok[i] and geo(Rv[i][None], R_gyro[i][None])[0] <= TAU for i in range(N)]))}
        rows[f"{key}_only"] = (np.where(ok, geo(Rv, R_gt), np.nan), np.where(ok, np.linalg.norm(tv * np.array([sc_ref[s] for s in seq])[:, None] - t_gt, axis=1), np.nan))
        for name, (gR, gt, gate) in {"rig_weights": (*W_RIG[key], True), "default": (0.5, 0.5, True), "nogate": (0.5, 0.5, False),
                                     "gammaR0_ungated": (0.0, W_RIG[key][1], False)}.items():
            r, t, u = blend(key, gR, gt, gate)
            rows[f"{key}_{name}"] = (r, t)
            out["damage"][f"{key}_{name}"] = {"used": u, "max_rot_increase_deg": float((r - e_gyro).max()), "max_trans_increase_mm": float((t - rows["inertial"][1]).max()),
                                              "p95_trans_increase_mm": float(np.percentile(t - rows["inertial"][1], 95))}
            if name == "rig_weights":
                r2, t2, _ = blend(key, gR, gt, gate, ref_scale=True)
                out["damage"][f"{key}_{name}"]["trans_mean_ref_scale"] = float(t2.mean())
        for name, (gR, gt, gate) in {"rig_weights_tlio": (*W_RIG[key], True), "gammaR0_ungated_tlio": (0.0, W_RIG[key][1], False)}.items():
            r, t, u = blend(key, gR, gt, gate, tsrc=t_tlio)
            rows[f"{key}_{name}"] = (r, t)
            out["damage"][f"{key}_{name}"] = {"used": u, "max_rot_increase_deg": float((r - e_gyro).max()),
                                              "max_trans_increase_mm": float((t - rows["inertial_tlio"][1]).max())}
    for k, (r, t) in rows.items():
        m = ~np.isnan(r)
        out["rows"][k] = summ(r[m], t[m])
        idx = np.arange(N)[::NONOVL]
        mm = m[idx]
        out["rows_non_overlapping"][k] = summ(r[idx][mm], t[idx][mm])
        out["per_sequence"][k] = {n: {"rot": float(np.nanmean(r[seq == n])), "trans": float(np.nanmean(t[seq == n]))} for n in names}
    out["per_sequence"]["N"] = {n: int((seq == n).sum()) for n in names}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("coverage", out["coverage"])
    for k, v in out["rows"].items():
        print(f"ROW {k:22s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f}  trans {v['trans_mean']:.3f}/{v['trans_p50']:.3f}/{v['trans_p95']:.3f}  N {v['N']}")
    print("NON-OVERLAPPING", {k: (round(v["rot_mean"], 3), round(v["trans_mean"], 3)) for k, v in out["rows_non_overlapping"].items()})
    print("DAMAGE", {k: (v["used"], round(v["max_rot_increase_deg"], 2), round(v["max_trans_increase_mm"], 1), round(v.get("trans_mean_ref_scale", float("nan")), 3)) for k, v in out["damage"].items()})


if __name__ == "__main__":
    main()
