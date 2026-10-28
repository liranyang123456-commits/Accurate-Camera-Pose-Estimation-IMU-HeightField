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
    """Inertial increment of the method (gyroscope rotation, GRU6D translation) and its gated fusion (Tables 4, 5, 18)."""
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


def addenda():
    """Session-averaged means, shrinkage control, ungated baseline, Holm adjustment, and time-offset diagnostics (Table 8 and Sec. 4.2)."""
    a = json.loads((HERE / "review_addenda_results.json").read_text(encoding="utf-8"))
    print("ADDENDA window share of the two largest sessions:", round(a["window_share"]["largest_two_fraction"], 3))
    for p in a["per_session"]:
        print(f"ADDENDA SESSION {p['session'][-6:]} N {p['N']:4d} delta {p['best_delta_s']:+.2f} PnP {p['reproj_rmse_median_px']:.2f} px | rot zero {p['zero_rot']:.2f} "
              f"gyro {p['gyro_rot']:.2f} gru6d {p['gru6d_rot']:.2f} | trans zero {p['zero_trans']:.2f} inertial {p['inertial_trans']:.2f} "
              f"vo {p['vo_validation_trans']:.2f} hf {p['hf_validation_trans']:.2f}")
    print("ADDENDA MACRO", {k: (round(v["rot"], 3), round(v["trans"], 3)) for k, v in a["macro_means"].items()})
    s = a["shrinkage"]
    print(f"ADDENDA SHRINK s_val {s['s_validation']} -> {s['test_trans_at_s_validation']:.3f} mm (oracle s {s['s_test_oracle']} -> {s['test_trans_at_s_test']:.3f}); "
          f"cosine mean {s['direction_cosine_mean']:.2f}, negative {s['fraction_negative_cosine']:.3f}; gains vs shrunk "
          + str({k: round(v, 3) for k, v in s["gain_vs_shrunk"].items()}))
    print("ADDENDA UNGATED", {k: (v["gamma_t_validation_ungated"], round(v["test_trans"], 3), v["n_used"]) for k, v in a["ungated_gammaR0"].items()})
    print("ADDENDA DAMAGE", {k: (round(v["max_increase_mm"], 1), round(v["p95_increase_mm"], 2), v["n_increase_gt5mm"]) for k, v in a["translation_damage"].items()})
    print("ADDENDA SWAP", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in a["translation_source_swap"].items()})
    print("ADDENDA CAUSAL", {k: (round(v["rot"], 3), round(v["trans"], 3), v["n_used"]) for k, v in a["causal_scale"].items()})
    print("ADDENDA HOLM session table", {k: round(v["p_holm"], 3) for k, v in a["holm_table_session_stats"].items()})
    print("ADDENDA HOLM visual", {b: {k: round(v, 3) for k, v in d.items()} for b, d in a["holm_visual_tables"].items()})
    t = a["time_offset"]
    print(f"ADDENDA OFFSET test sessions within 0.05 s of -0.50: {t['n_sessions_delta_within_0.05_of_minus0.50']}/11; oracle per-session delta pooled "
          f"{t['oracle_session_delta_pooled_mean']:.3f} deg; consecutive-frame-pair estimate {t['consecutive_frame_pairs']['best_delta_s']} s "
          f"({t['consecutive_frame_pairs']['n_pairs']} pairs, mean dt {t['consecutive_frame_pairs']['mean_dt_s']:.3f} s)")


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
    addenda()
    for name in ("miti_target_regime_results.json", "miti_target_regime_results_stride10.json", "miti_fusion_results.json", "tumvi_framerate_results.json"):
        f = HERE / name
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if "rows" in d and "gyro" in d["rows"]:
            print("MITI seq01", name, {k: round(v["mean"], 3) for k, v in d["rows"].items() if isinstance(v, dict)}, "gate accept", round(d["gate_accept_fraction"], 3))
        if "encoders_test_trans_mm" in d:
            print("MITI nine sequences N", d["N"], "encoders trans mm", {k: round(v, 3) for k, v in d["encoders_test_trans_mm"].items()})
            for k, v in d["rows"].items():
                print(f"MITI {k:28s} rot {v['rot_mean']:.3f}/{v['rot_p50']:.3f}/{v['rot_p95']:.3f}  trans {v['trans_mean']:.3f}/{v['trans_p50']:.3f}/{v['trans_p95']:.3f}")
        if "runs" in d:
            for k, v in d["runs"].items():
                print("TUMVI", k, round(v["frame_rate_hz"], 1), "Hz", {s: (v[s].get("num_est_poses"), round(v[s].get("ate_se3_rmse_m", float("nan")), 3))
                                                                   for s in ("orbslam3_mono_inertial", "vins_fusion_stereo_inertial") if s in v})


if __name__ == "__main__":
    main()
