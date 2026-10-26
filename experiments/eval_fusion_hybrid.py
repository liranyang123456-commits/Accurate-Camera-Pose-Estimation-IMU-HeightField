#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gated fusion with the inertial increment of the method: gyroscope rotation and GRU6D translation.

The rotation is the time-aligned gyroscope increment R_CI B^T R_CI^T with the offset and R_CI of
eval_gyro_time_alignment.py (49 training and validation sessions); the translation is that of the evaluated GRU6D
(seed 0) trained on the aligned windows. The rotation and translation weights of each consistency-gated front end are
selected on the 13 validation sessions by a grid search with step 0.05 and applied unchanged to the 11 test sessions;
the closed-form weights of Proposition 3(ii) on the gated validation increments are reported for reference.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from build_fixed_seq_dataset_from_session import load_imu_stream_features  # noqa: E402

import eval_fixedtest_aligned as ev  # noqa: E402
import eval_fusion_weights_validation as fw  # noqa: E402
import eval_gyro_time_alignment as ga  # noqa: E402

TAU = 3.0
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
OUT = HERE / "fusion_hybrid_results.json"


def add_gyro(recs, delta, X):
    by = {}
    for r in recs:
        by.setdefault(r["session"], []).append(r)
    for n, rs in by.items():
        sd = fw.SESS / n
        t, feat, _ = load_imu_stream_features(str(sd / "imu_stream.csv"))
        meta = list(csv.DictReader(open(sd / ev.DS / "meta.csv", encoding="utf-8")))
        B = ga.body_increment(ga.gyro_track(t, feat[:, 3:6]), np.array([float(m["t_start"]) for m in meta]),
                              np.array([float(m["t_end"]) for m in meta]), delta)
        assert len(B) == len(rs), n
        for r, b in zip(rs, B):
            r["R_gyro"] = X @ b.T @ X.T


def blend(recs, key, gR, gt, scales, tmode="inertial", tau=TAU, gate=True):
    rot, tr, used = [], [], 0
    for r in recs:
        R, t = r["R_gyro"], r["gru"][0][1]
        if key and r[key + "_ok"] and (not gate or ev.geo(r["R_" + key], R) <= tau):
            used += 1
            if gR > 0:
                R = ev.slerp(R, r["R_" + key], gR)
            s = r["s_" + key] if tmode == "ref" else scales[r["session"]]
            t = gt * r["t_" + key] * s + (1 - gt) * t
        rot.append(ev.geo(R, r["R_gt"]))
        tr.append(float(np.linalg.norm(t - r["t_gt"])))
    return np.asarray(rot), np.asarray(tr), used


def summ(r, t, used=None, base_r=None):
    out = {"rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
           "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)),
           "hit_rot5": float(np.mean(r <= 5)), "hit_trans20": float(np.mean(t <= 20)), "N": int(r.size)}
    if used is not None:
        out["n_visual_used"] = int(used)
    if base_r is not None:
        out["max_excess_deg"] = float((r - base_r).max())
    return out


def select(vrecs, key):
    sc = ev.inertial_scales(vrecs, key, 0)
    G = [r for r in vrecs if r[key + "_ok"] and ev.geo(r["R_" + key], r["R_gyro"]) <= TAU]
    eR = lambda f: np.array([ev.geo(f(r), r["R_gt"]) for r in G]) ** 2  # noqa: E731
    et = lambda f: np.array([np.linalg.norm(f(r) - r["t_gt"]) for r in G]) ** 2  # noqa: E731
    mi_R, mv_R = eR(lambda r: r["R_gyro"]).mean(), eR(lambda r: r["R_" + key]).mean()
    mi_t, mv_t = et(lambda r: r["gru"][0][1]).mean(), et(lambda r: r["t_" + key] * sc[r["session"]]).mean()
    rot = {float(g): float(blend(vrecs, key, g, 0.0, sc)[0].mean()) for g in GRID}
    tr = {float(g): float(blend(vrecs, key, 0.0, g, sc)[1].mean()) for g in GRID}
    return {"n_gated": len(G), "n_windows": len(vrecs), "closed_form_gamma_R": float(mi_R / (mi_R + mv_R)),
            "closed_form_gamma_t": float(mi_t / (mi_t + mv_t)), "gamma_R": min(rot, key=rot.get), "gamma_t": min(tr, key=tr.get),
            "validation_rot_by_gamma": rot, "validation_trans_by_gamma": tr}


def main():
    al = json.loads((HERE / "gyro_time_alignment_results.json").read_text(encoding="utf-8"))
    g = al["variants"]["gyro_corrected_convention_offset"]
    delta, X = float(g["delta_s"]), np.asarray(g["R_CI"])
    val = json.loads((HERE / "validation_sessions.json").read_text(encoding="utf-8"))["validation"]
    vrecs = fw.records(val, fw.gru6d_predict(val))
    add_gyro(vrecs, delta, X)
    trecs, _ = ev.build_records()
    sess = np.array([r["session"] for r in trecs])
    names = list(dict.fromkeys(sess.tolist()))
    br, bt, _ = blend(trecs, None, 0, 0, None)
    out = {"protocol": __doc__.strip().splitlines()[0], "tau": TAU, "delta_s": delta, "weights": {}, "rows": {}, "stats": {},
           "sensitivity": [], "validation_inertial": {}}
    vr, vt, _ = blend(vrecs, None, 0, 0, None)
    out["validation_inertial"] = summ(vr, vt)
    out["rows"]["inertial"] = summ(br, bt)
    scales = {k: ev.inertial_scales(trecs, k, 0) for k in ("vo", "hf", "slam")}
    for key in ("vo", "hf"):
        w = select(vrecs, key)
        out["weights"][key] = w
        for name, (gR, gt, gate) in {"default": (0.5, 0.5, True), "validation": (w["gamma_R"], w["gamma_t"], True),
                                     "nogate": (0.5, 0.5, False)}.items():
            r, t, u = blend(trecs, key, gR, gt, scales[key], gate=gate)
            out["rows"][f"{key}_{name}"] = summ(r, t, u, br)
            out["rows"][f"{key}_{name}"].update({"gamma_R": gR, "gamma_t": gt})
            if name != "nogate":
                out["rows"][f"{key}_{name}"]["trans_mean_ref_scale"] = float(blend(trecs, key, gR, gt, scales[key], "ref", gate=gate)[1].mean())
                out["stats"][f"{key}_{name}_rot"] = ev.session_stats(br - r, sess, names)
                out["stats"][f"{key}_{name}_trans"] = ev.session_stats(bt - t, sess, names)
    r, t, u = blend(trecs, "slam", 0.5, 0.5, scales["slam"], gate=False)
    out["rows"]["slam_default"] = summ(r, t, u, br)
    out["rows"]["slam_default"]["trans_mean_ref_scale"] = float(blend(trecs, "slam", 0.5, 0.5, scales["slam"], "ref", gate=False)[1].mean())
    out["stats"]["slam_default_rot"] = ev.session_stats(br - r, sess, names)
    out["stats"]["slam_default_trans"] = ev.session_stats(bt - t, sess, names)
    for tau in (2.0, 3.0, 5.0, 10.0):
        row = {"tau": tau}
        for gam in (0.1, 0.3, 0.5, 0.7):
            r, t, u = blend(trecs, "vo", gam, gam, scales["vo"], tau=tau)
            row[str(gam)] = {"rot_mean": float(r.mean()), "trans_mean": float(t.mean()), "n_used": u, "max_excess": float((r - br).max())}
        out["sensitivity"].append(row)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("validation inertial:", {k: round(v, 3) for k, v in out["validation_inertial"].items()})
    for k, w in out["weights"].items():
        print(f"{k}: gated {w['n_gated']}/{w['n_windows']} grid gamma_R {w['gamma_R']} gamma_t {w['gamma_t']} | closed form "
              f"{w['closed_form_gamma_R']:.3f} / {w['closed_form_gamma_t']:.3f}")
    for k, v in out["rows"].items():
        print(f"ROW {k:15s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f} trans {v['trans_mean']:.3f}/{v['trans_p50']:.3f}/"
              f"{v['trans_p95']:.3f} hit {v['hit_rot5']:.3f}/{v['hit_trans20']:.3f}"
              + (f" used {v['n_visual_used']}" if "n_visual_used" in v else "")
              + (f" max_excess {v['max_excess_deg']:.2f}" if "max_excess_deg" in v else "")
              + (f" ref {v['trans_mean_ref_scale']:.3f}" if "trans_mean_ref_scale" in v else ""))
    for k, v in out["stats"].items():
        print(f"STAT {k:22s} {v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {v['sessions_better']}/11 p {v['p']:.4f}")
    for row in out["sensitivity"]:
        print("SENS", row["tau"], {gk: (round(gv["rot_mean"], 3), round(gv["trans_mean"], 3), gv["n_used"], round(gv["max_excess"], 2))
                                    for gk, gv in row.items() if gk != "tau"})


if __name__ == "__main__":
    main()
