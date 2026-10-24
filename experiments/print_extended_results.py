#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the 11-session fusion results, the learned inertial encoders, the visual benchmarks, the runtime,
and the transfer checks from the bundled JSON files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = {"heightfield": "Ours (height field)", "static": "Static (reference)", "vggt": "VGGT", "endodac": "EndoDAC",
       "cut3r": "CUT3R", "reloc3r": "Reloc3r", "afsfm": "AF-SfMLearner", "endofast3r": "Endo-FASt3R", "pi3": "Pi3",
       "droid": "DROID-SLAM", "dust3r": "DUSt3R (32 frames)", "endo3r": "Endo3R (32 frames)", "mast3r": "MASt3R (32 frames)"}


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _gain(g: dict) -> str:
    lo, hi = g["cluster_ci95"]
    return (f"gain {g['window_mean_gain']:+.3f}, CI [{lo:.3f}, {hi:.3f}], sessions {g['sessions_improved']}/{g['n_sessions']}, "
            f"p={g['session_wilcoxon_p']:.4f}")


def fusion() -> None:
    print("\n--- [A] 11 held-out sessions (N=2312 windows of 0.5 s): inertial, visual, and gated estimators ---")
    vi = _load("combined_vi_and_stats.json")
    for k in ("gru6d", "slam_blend", "vo_gated"):
        r = vi[k]
        print(f"  {r['name']:58s} rot {r['rot_mean']:.3f} deg | trans {r['trans_mean']:.3f} mm | Hit@5deg {r['hit_rot5']:.3f}")
    j = _load("joint_hf_gru6d_results.json")
    names = {"zero_rot": "Zero rotation (reference)", "hf_only": "Height field, visual-only rotation",
             "hf_only_ref": "Height field, visual-only (reference scale)", "hf_gated_rot": "Gated height field + GRU6D (rotation only)",
             "hf_gated_se3": "Gated height field + GRU6D (SE(3), inertial scale)",
             "hf_gated_se3_ref": "Gated height field + GRU6D (SE(3), reference scale)"}
    for k, lab in names.items():
        r = j["streams"][k]
        print(f"  {lab:58s} rot {r['rot_mean']:.3f} deg | trans {r['trans_mean']:.3f} mm | N={r['N']} | visual used {r['n_visual_used']}")
    print(f"    session level vs. GRU6D, gated height field: rot {_gain(j['streams']['hf_gated_rot']['rot_gain_vs_gru6d'])}")
    for k in ("hf_gated_se3", "hf_gated_se3_ref"):
        print(f"    session level vs. GRU6D, {k}: trans {_gain(j['streams'][k]['trans_gain_vs_gru6d'])}")

    ss = _load("session_cluster_stats.json")
    print("\n--- [A2] session-level statistics of the ORB-VO and ORB-SLAM3 blends (interval resamples sessions) ---")
    for k, v in ss["rotation_deg"].items():
        lo, hi = v["cluster_bootstrap_ci95_mean_diff"]
        print(f"  rot {k:46s} gain {v['window_mean_diff']:.3f} deg, CI [{lo:.3f}, {hi:.3f}], "
              f"sessions {v['n_sessions_proposed_better']}/{v['n_sessions']}, p={v['session_wilcoxon_p']:.4f}")
    for k, v in ss["translation_mm"].items():
        if k == "gru6d_mean":
            continue
        g = v["gru6d_minus_fused"]
        lo, hi = g["cluster_bootstrap_ci95_mean_diff"]
        print(f"  trans {k:40s} mean {v['mean']:.3f} mm, gain CI [{lo:.3f}, {hi:.3f}], "
              f"sessions {g['n_sessions_proposed_better']}/{g['n_sessions']}, p={g['session_wilcoxon_p']:.4f}")


def encoders() -> None:
    p = ROOT / "inertial_baselines" / "inertial_baselines_summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    ref = s.get("session_reference", "GRU6D")
    print(f"\n--- [A3] learned inertial encoders, same heads, loss, split, and schedule (session tests vs. {ref}) ---")
    for k, v in s.items():
        if not isinstance(v, dict):
            continue
        r, t = v["rot_mean"], v["trans_mean"]
        line = f"  {k:14s} seeds {v['n_seeds']} | rot {sum(r) / len(r):.3f} deg | trans {sum(t) / len(t):.3f} mm | {v['params_M']:.2f} M params"
        c = v.get("vs_retrained") or (v["vs_checkpoint"] if ref == "GRU6D" else None)
        if c:
            line += (f" | rot gain {c['rot_gain']:+.3f} ({c['rot_sessions_better']}/11, p={c['rot_p']:.3f})"
                     f" | trans gain {c['trans_gain']:+.3f} ({c['trans_sessions_better']}/11, p={c['trans_p']:.3f})")
        print(line)


def visual() -> None:
    print("\n--- [B] Sim(3) ATE (mm) of the height-field tracker and the compared methods, 64 uniform frames ---")
    for name, title in (("real18", "18 real 4K-referenced segments"), ("sim92", "92 simulated sequences"),
                        ("c3vd10", "10 C3VD phantom sequences")):
        s = _load(f"hf_benchmarks/{name}_reeval.json")["summary"]
        print(f"  {title}")
        for k, v in s.items():
            if k == "orbslam3_harness":
                print(f"    ORB-SLAM3 (harness): tracked {v['tracked']}, failed {v['failed']}, Sim(3) on tracked {v['sim3_mean_on_tracked']:.2f} mm")
                continue
            if name == "sim92":
                tr, orb = v["translating"], v["orbital"]
                extra = f" | ours better {tr['hf_better_on']}/71, p={tr['wilcoxon_p']:.3g}" if "hf_better_on" in tr else ""
                print(f"    {LAB.get(k, k):24s} translating (71): {tr['sim3_mean']:.2f} / {tr['sim3_median']:.2f}"
                      f" | orbital (21) RPE rot {orb['rpe_r_mean']:.3f} deg{extra}")
            elif "sim3_mean" in v:
                extra = f" | ours better {v['hf_better_on']}/{v['paired_n']}, p={v['wilcoxon_p']:.3g}" if "hf_better_on" in v else ""
                print(f"    {LAB.get(k, k):24s} {v['sim3_mean']:.2f} / {v['sim3_median']:.2f} (tracked {v['tracked']}/{v['total']}){extra}")


def controlled() -> None:
    p = ROOT / "controlled_fusion_sim_results.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    tau = d["tau_deg"]
    print(f"\n--- [B2] controlled fusion with SIMULATED inertial increments (tau = {tau} deg, {d['seeds']} noise seeds) ---")
    for name, title in (("sim92", "92 simulated sequences"), ("real18", "18 real segments")):
        b = d[name]
        print(f"  {title}: height field alone {b['visual_only']['rot_all']:.3f} deg ({b['n_increments']} increments)")
        for sr, r in b["results"].items():
            best = min(d["gamma_grid"], key=lambda g: r[f"g{g}_t{tau}"]["rot_all"][0])
            print(f"    sigma_R={sr:>3} deg: inertial {r['imu']['rot_all'][0]:.3f} | blend {r[f'g0.5_t{tau}']['rot_all'][0]:.3f} "
                  f"(accepted {100 * r[f'g0.5_t{tau}']['accepted'][0]:.0f}%, max excess {r[f'g0.5_t{tau}']['max_excess'][0]:.2f}) | "
                  f"no gate {r['g0.5_tinf']['rot_all'][0]:.3f} (max excess {r['g0.5_tinf']['max_excess'][0]:.2f}) | "
                  f"gamma*={r['gamma_star_gated'][0]:.2f} -> {r['gstar_gated']['rot_all'][0]:.3f} | best grid gamma={best} | "
                  f"ATE {r['imu']['ate_trans'][0]:.2f} / {r[f'g0.5_t{tau}']['ate_trans'][0]:.2f} / {r['g0.5_rot_only']['ate_trans'][0]:.2f} mm")


def runtime() -> None:
    r = _load("runtime_results.json")
    print(f"\n--- [C] runtime on {r['frames']} frames ({r['machine']['processor']}) ---")
    for row in r["tracker"]:
        print(f"  tracker {row['resolution']:8s} one thread: {row['total_ms']:.1f} ms/frame "
              f"(front end {row['front_end_ms']:.1f}, registration {row['registration_ms']:.1f}); heap peak {row['python_heap_peak_mb']:.1f} MB")
    m = _load("runtime_multithread.json")
    for row in m["rows"]:
        print(f"  tracker 640 px, {row['max_pts']} points, {m['threads']} threads: {row['total_ms']:.1f} ms/frame")
    g = r["gru6d"]
    print(f"  GRU6D one thread: {g['ms_per_window']:.2f} ms/window, {g['params'] / 1e6:.2f} M parameters")


def transfer() -> None:
    print("\n--- [D] transfer and VIO diagnostics ---")
    cc = _load("gru6d_cross_capture/cross_capture_metrics_12x9.json")
    pooled = cc.get("pooled") or cc.get("overall") or {}
    print("  frozen GRU6D on the second rig (12x9 target):", {k: pooled[k] for k in pooled if k in ("N", "rot_deg_mean", "trans_mean")})
    cov = _load("r13_coverage/coverage_metrics.json")
    print(f"  Kalibr capture: ORB-SLAM3 mono {cov['n_frames_mono_matched']}/{cov['n_frames']}, mono-inertial "
          f"{cov['n_frames_mi_matched']}/{cov['n_frames']}, IMU windows {cov['n_imu_windows']}/{cov['n_frames']}")
    hard = _load("cholecseg8k_hard/protocol.json")
    print("  CholecSeg8k subset counts:", hard["subset_counts"])


def main() -> None:
    fusion()
    encoders()
    visual()
    controlled()
    runtime()
    transfer()


if __name__ == "__main__":
    main()
