#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print every number of the revised inertial and visual-inertial text from the result files of the aligned runs."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "inertial_baselines_aligned"


def calibration():
    g = json.loads((HERE / "gyro_time_alignment_results.json").read_text(encoding="utf-8"))
    po = g["per_session_offset"]
    print("LATENCY delta %.2f s (per-session median %.2f s, IQR %s)" % (g["variants"]["gyro_corrected_convention_offset"]["delta_s"],
                                                                        po["median_s"], po["iqr_s"]))
    for k, v in g["variants"].items():
        print(f"GYRO/QUAT {k:40s} mean {v['mean']:.3f} median {v['median']:.3f}")
    c = json.loads((HERE / "inertial_classical_aligned_results.json").read_text(encoding="utf-8"))
    for k, v in c["rows"].items():
        print(f"CLASSICAL {k:12s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f} trans {v['trans_mean']:.3f}/"
              f"{v['trans_p50']:.3f}/{v['trans_p95']:.3f} hit {v['hit_rot5']:.3f}/{v['hit_trans20']:.3f}")
    m = json.loads((HERE / "orbslam3_mono_inertial_aligned_coverage.json").read_text(encoding="utf-8"))
    print("ORB-SLAM3 MONO-INERTIAL (recalibrated): trajectories", m["n_with_traj"], "/ 11, RPE windows", m["n_rpe_windows"])
    f = json.loads((HERE / "fusion_weights_validation_results.json").read_text(encoding="utf-8"))
    for k, w in f["weights"].items():
        print(f"VALIDATION {k}: closed form gamma_R {w['gamma_R']:.3f} gamma_t {w['gamma_t']:.3f}; grid gamma_R {w['grid_best_gamma_R']} "
              f"gamma_t {w['grid_best_gamma_t']}")
    for k, v in f["test_grid_weights"].items():
        a, b = v["rot_vs_gru6d"], v["trans_vs_gru6d"]
        print(f"TEST validation weights {k}: rot {v['rot_mean']:.3f} ({a['gain']:+.3f}, {a['sessions_better']}/11, p {a['p']:.3f}) "
              f"trans {v['trans_mean']:.3f} ({b['gain']:+.3f}, {b['sessions_better']}/11, p {b['p']:.3f}) max excess {v['max_excess']:.2f}")
    sens = HERE / "vi_gate_sensitivity_inertial_scale.json"
    if sens.exists():
        for row in json.loads(sens.read_text(encoding="utf-8")):
            cells = " | ".join(f"g {g}: {row[g][0]:.3f} deg {row[g][1]:.3f} mm" for g in ("0.1", "0.3", "0.5", "0.7"))
            print(f"SENS (inertial scale) tau {row['tau']:4.1f} used {row['0.5'][2]:4d} | {cells} | max excess at 0.5: {row['0.5'][3]:.2f}")


def hybrid():
    """Inertial increment of the method (gyroscope rotation, GRU6D translation) and its gated fusion (Tables 4, 5, 15)."""
    h = json.loads((HERE / "fusion_hybrid_results.json").read_text(encoding="utf-8"))
    for k, w in h["weights"].items():
        print(f"HYBRID weights {k}: grid gamma_R {w['gamma_R']} gamma_t {w['gamma_t']} | closed form {w['closed_form_gamma_R']:.3f} / "
              f"{w['closed_form_gamma_t']:.3f} ({w['n_gated']}/{w['n_windows']} gated validation windows)")
    for k, x in h["rows"].items():
        print(f"HYBRID ROW {k:15s} rot {x['rot_mean']:.3f}/{x['rot_p50']:.3f}/{x['rot_p95']:.3f} trans {x['trans_mean']:.3f}/"
              f"{x['trans_p50']:.3f}/{x['trans_p95']:.3f} hit {x['hit_rot5']:.3f}/{x['hit_trans20']:.3f}"
              + (f" used {x['n_visual_used']}" if "n_visual_used" in x else "")
              + (f" max excess {x['max_excess_deg']:.2f}" if "max_excess_deg" in x else "")
              + (f" ref scale {x['trans_mean_ref_scale']:.3f}" if "trans_mean_ref_scale" in x else ""))
    for k, x in h["stats"].items():
        print(f"HYBRID STAT {k:22s} {x['gain']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] {x['sessions_better']}/11 p {x['p']:.4f}")
    for row in h["sensitivity"]:
        cells = " | ".join(f"g {g}: {row[g]['rot_mean']:.3f} deg {row[g]['trans_mean']:.3f} mm" for g in ("0.1", "0.3", "0.5", "0.7"))
        print(f"HYBRID SENS tau {row['tau']:4.1f} used {row['0.5']['n_used']:4d} | {cells} | max excess at 0.5: {row['0.5']['max_excess']:.2f}")


def main():
    calibration()
    runs = {}
    for f in sorted(D.glob("inertial_baselines_results_s*.json")):
        runs.update(json.loads(f.read_text(encoding="utf-8"))["runs"])
    g0 = runs["gru6d_s0"]
    z = np.load(D / "pred_errors_gru6d_s0.npz")
    r = np.concatenate([z[k] for k in z.files if k.endswith("__rot")])
    t = np.concatenate([z[k] for k in z.files if k.endswith("__trans")])
    v = g0["validation"]
    print("TABLE 3 validation: rot %.2f+-%.2f p50/p90 %.2f/%.2f max %.1f | trans %.2f+-%.2f (%.2f/%.2f)" %
          (v["rot_mean"], v["rot_sd"], v["rot_p50"], v["rot_p90"], v["rot_max"], v["trans_mean"], v["trans_sd"], v["trans_p50"], v["trans_p90"]))
    print("TABLE 3 test:       rot %.2f+-%.2f p50/p90 %.2f/%.2f max %.1f | trans %.2f+-%.2f (%.2f/%.2f)" %
          (r.mean(), r.std(), np.median(r), np.percentile(r, 90), r.max(), t.mean(), t.std(), np.median(t), np.percentile(t, 90)))
    print("best val loss per run:", {k: round(x["best_val_loss"], 4) for k, x in runs.items()})
    for kind in ("ronin_resnet", "ronin_lstm", "tlio_resnet", "gru6d"):
        rr = [runs[f"{kind}_s{s}"] for s in range(3) if f"{kind}_s{s}" in runs]
        rot = np.array([x["rot_mean"] for x in rr])
        tr = np.array([x["trans_mean"] for x in rr])
        print(f"ENC {kind:13s} n={len(rr)} rot {rot.mean():.3f}+-{rot.std(ddof=1) if len(rot) > 1 else 0:.3f} "
              f"trans {tr.mean():.3f}+-{tr.std(ddof=1) if len(tr) > 1 else 0:.3f} seeds rot {np.round(rot, 3).tolist()}")
    summ = D / "inertial_baselines_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text(encoding="utf-8"))
        for k, x in s.items():
            if isinstance(x, dict) and "vs_retrained" in x:
                c = x["vs_retrained"]
                print(f"SESS {k:13s} vs GRU6D rot {c['rot_gain']:+.3f} [{c['rot_ci'][0]:+.3f},{c['rot_ci'][1]:+.3f}] {c['rot_sessions_better']}/11 "
                      f"p {c['rot_p']:.3f} | trans {c['trans_gain']:+.3f} [{c['trans_ci'][0]:+.3f},{c['trans_ci'][1]:+.3f}] "
                      f"{c['trans_sessions_better']}/11 p {c['trans_p']:.3f}")
        if "gru6d_vs_gyro" in s:
            c = s["gru6d_vs_gyro"]
            print(f"SESS GRU6D(3 seeds) vs gyro rot {c['rot_gain']:+.3f} [{c['rot_ci'][0]:+.3f},{c['rot_ci'][1]:+.3f}] {c['rot_sessions_better']}/11 p {c['rot_p']:.3f}")
    R = json.loads((HERE / "fixedtest_aligned_results.json").read_text(encoding="utf-8"))
    for k, x in R["rows"].items():
        print(f"ROW {k:24s} rot {x['rot_mean']:.3f}/{x['rot_p50']:.3f}/{x['rot_p95']:.3f} trans {x['trans_mean']:.3f}/{x['trans_p50']:.3f}/"
              f"{x['trans_p95']:.3f} hit {x['hit_rot5']:.3f}/{x['hit_trans20']:.3f} N {x['N']}" + (f" used {x['n_visual_used']}" if "n_visual_used" in x else ""))
    for k, x in R["stats"].items():
        print(f"STAT {k:34s} {x['gain']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] {x['sessions_better']}/11 p {x['p']:.4f}")
    print("BOUND", {k: (round(x["max_excess_deg"], 2), x["bound_deg"]) for k, x in R["bound"].items()})
    print("SUBSET", R["subset"])
    print("SEEDS", R["seed_range"])
    for x in R["sensitivity"]:
        print(f"SENS tau {x['tau']:4.1f} g {x['gamma']:.1f} used {x['n_used']:4d} rot {x['rot_mean']:.3f} ({x['rot_p50']:.3f}) trans {x['trans_mean']:.3f} "
              f"hit {x['hit_rot5']:.3f}/{x['hit_trans20']:.3f} max_excess {x['max_excess']:.2f}")
    hybrid()


if __name__ == "__main__":
    main()
