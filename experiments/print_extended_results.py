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
    print("\n--- [D] VIO diagnostics ---")
    cov = _load("r13_coverage/coverage_metrics.json")
    print(f"  Kalibr capture: ORB-SLAM3 mono {cov['n_frames_mono_matched']}/{cov['n_frames']}, mono-inertial "
          f"{cov['n_frames_mi_matched']}/{cov['n_frames']}, IMU windows {cov['n_imu_windows']}/{cov['n_frames']}")
    hard = _load("cholecseg8k_hard/protocol.json")
    print("  CholecSeg8k subset counts:", hard["subset_counts"])


def main() -> None:
    from experiments.print_aligned_numbers import main as aligned
    print("\n--- [A] 11 held-out sessions after the camera-IMU time alignment: inertial, visual, and gated estimators ---")
    aligned()
    visual()
    controlled()
    runtime()
    transfer()


if __name__ == "__main__":
    main()
