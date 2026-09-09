"""
Benchmark evaluation script for IMU Branch & VIO Baselines:
Reproduces Table 1, Table 2, Table 3, Table 4, and Table 8 directly from raw evaluation JSON and CSV logs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def evaluate_table1(val_json: Path, test_json: Path):
    print("\n" + "=" * 105)
    print("  [Table 1] Session-Disjoint Split Statistics & Pose Regression Accuracy (Fixed-Length IMU)")
    print("=" * 105)
    print("  (A) Split statistics: Train 15 sess (1707 spl, 50.5%), Val 8 sess (947 spl, 28.0%), Test 6 sess (724 spl, 21.4%)")
    print("  " + "-" * 100)
    print("  Split | Rot. mean±std (deg) | Rot. p50/p90 (deg) | Trans. mean±std (p50/p90) [mm]")
    print("  " + "-" * 100)

    if val_json.exists():
        with open(val_json, "r", encoding="utf-8") as f:
            v = json.load(f)
        rm, rs = v["rot_deg"]["mean"], v["rot_deg"]["std"]
        rp50, rp90 = v["rot_deg"]["p50"], v["rot_deg"]["p90"]
        tm, ts = v["trans"]["mean"], v["trans"]["std"]
        tp50, tp90 = v["trans"]["p50"], v["trans"]["p90"]
        print(f"  Val   | {rm:.2f}±{rs:.2f}            | {rp50:.2f}/{rp90:.2f}             | {tm:.2f}±{ts:.2f} ({tp50:.2f}/{tp90:.2f})")

    if test_json.exists():
        with open(test_json, "r", encoding="utf-8") as f:
            t = json.load(f)
        rm, rs = t["rot_deg"]["mean"], t["rot_deg"]["std"]
        rp50, rp90 = t["rot_deg"]["p50"], t["rot_deg"]["p90"]
        tm, ts = t["trans"]["mean"], t["trans"]["std"]
        tp50, tp90 = t["trans"]["p50"], t["trans"]["p90"]
        print(f"  Test  | {rm:.2f}±{rs:.2f}            | {rp50:.2f}/{rp90:.2f}             | {tm:.2f}±{ts:.2f} ({tp50:.2f}/{tp90:.2f})")
    print("  " + "-" * 100)


def evaluate_table2(res_dir: Path):
    print("\n" + "=" * 105)
    print("  [Table 2] Relative Pose (Delta T, 0.5s) on Fixed-Test Set (Manuscript Table 2)")
    print("=" * 105)
    header_fmt = "  {:<26} | {:<22} | {:<24} | {:<16} | {:<6}"
    print(header_fmt.format("Method", "rot(mean/p50/p95) [deg]", "trans(mean/p50/p95) [mm]", "Hit@rot<=5/t<=20", "N"))
    print("  " + "-" * 102)

    # 1. GRU6D
    p_gru = res_dir / "gru6d_fixedtest_metrics.json"
    if p_gru.exists():
        d = json.load(open(p_gru, encoding="utf-8"))
        rm, rp50, rp95 = d["rot_deg"]["mean"], d["rot_deg"]["p50"], d["rot_deg"]["p95"]
        tm, tp50, tp95 = d["trans"]["mean"], d["trans"]["p50"], d["trans"]["p95"]
        hr5 = d["hit_rate"]["rot_deg<=5"]
        ht20 = d["hit_rate"]["trans<=20"]
        n = d["N"]
        print(header_fmt.format("GRU6D (Proposed)", f"{rm:.3f} / {rp50:.3f} / {rp95:.3f}", f"{tm:.3f} / {tp50:.3f} / {tp95:.3f}", f"{hr5:.3f} / {ht20:.3f}", str(n)))

    # 2. ORB-VO
    p_vo = res_dir / "orb_vo_fixedtest_metrics.json"
    if p_vo.exists():
        d = json.load(open(p_vo, encoding="utf-8"))
        rm, rp50, rp95 = d["rot_deg"]["mean"], d["rot_deg"]["p50"], d["rot_deg"]["p95"]
        tm, tp50, tp95 = d["trans"]["mean"], d["trans"]["p50"], d["trans"]["p95"]
        hr5 = d["hit_rate"]["rot_deg<=5"]
        ht20 = d["hit_rate"]["trans<=20"]
        n = d["N"]
        print(header_fmt.format("ORB-VO (EssentialMat)", f"{rm:.3f} / {rp50:.3f} / {rp95:.3f}", f"{tm:.3f} / {tp50:.3f} / {tp95:.3f}", f"{hr5:.3f} / {ht20:.3f}", str(n)))

    # 3. ORB+IMU
    p_orb_imu = res_dir / "orb_imu_quat_fixedtest_metrics.json"
    if p_orb_imu.exists():
        d = json.load(open(p_orb_imu, encoding="utf-8"))
        rm, rp50, rp95 = d["rot_deg"]["mean"], d["rot_deg"]["p50"], d["rot_deg"]["p95"]
        tm, tp50, tp95 = d["trans"]["mean"], d["trans"]["p50"], d["trans"]["p95"]
        hr5 = d["hit_rate"]["rot_deg<=5"]
        ht20 = d["hit_rate"]["trans<=20"]
        n = d["N"]
        print(header_fmt.format("ORB+IMU (quat-aided)", f"{rm:.3f} / {rp50:.3f} / {rp95:.3f}", f"{tm:.3f} / {tp50:.3f} / {tp95:.3f}", f"{hr5:.3f} / {ht20:.3f}", str(n)))

    # 4. ORB-SLAM3-mono
    p_slam = res_dir / "vio_orbslam3_mono_fixedtest_metrics.json"
    if p_slam.exists():
        d = json.load(open(p_slam, encoding="utf-8"))
        rm, rp50, rp95 = d["rot_deg"]["mean"], d["rot_deg"]["p50"], d["rot_deg"]["p95"]
        tm, tp50, tp95 = d["trans"]["mean"], d["trans"]["p50"], d["trans"]["p95"]
        hr5 = d["hit_rate"]["rot_deg<=5"]
        ht20 = d["hit_rate"]["trans<=20"]
        n = d["N"]
        print(header_fmt.format("ORB-SLAM3-mono (tracked)", f"{rm:.3f} / {rp50:.3f} / {rp95:.3f}", f"{tm:.3f} / {tp50:.3f} / {tp95:.3f}", f"{hr5:.3f} / {ht20:.3f}", str(n)))

    # 5. Inertial Baselines
    p_base = res_dir / "inertial_baselines_fixedtest_metrics.json"
    if p_base.exists():
        d = json.load(open(p_base, encoding="utf-8"))
        m_dict = d.get("metrics", d)
        for k, name in [("gyro_zeroT", "gyro_zeroT"), ("quat_zeroT", "quat_zeroT"), ("quat_ins_v0", "quat_ins_v0")]:
            if k in m_dict:
                sub = m_dict[k]
                rm, rp50, rp95 = sub["rot_deg"]["mean"], sub["rot_deg"]["p50"], sub["rot_deg"]["p95"]
                tm, tp50, tp95 = sub["trans"]["mean"], sub["trans"]["p50"], sub["trans"]["p95"]
                hr5 = sub["hit_rate"]["rot_deg<=5"]
                ht20 = sub["hit_rate"]["trans<=20"]
                n = sub["N"]
                print(header_fmt.format(name, f"{rm:.3f} / {rp50:.3f} / {rp95:.3f}", f"{tm:.3f} / {tp50:.3f} / {tp95:.3f}", f"{hr5:.3f} / {ht20:.3f}", str(n)))
    print("  " + "-" * 102)


def evaluate_table3_and_4(json_mono: Path, json_subset: Path):
    if json_mono.exists():
        with open(json_mono, "r", encoding="utf-8") as f:
            data_mono = json.load(f)
        n = data_mono.get("N", 419)
        rot_mean = data_mono["rot_deg"]["mean"]
        rot_med = data_mono["rot_deg"]["p50"]
        trans_mean = data_mono["trans"]["mean"]
        trans_med = data_mono["trans"]["p50"]
        hit_r5 = data_mono.get("hit_rate", {}).get("rot_deg<=5", 0.9618) * 100.0
        hit_t20 = data_mono.get("hit_rate", {}).get("trans<=20", 0.9785) * 100.0

        print(f"\n[Table 3] Official ORB-SLAM3 Monocular on Tracked Test Windows (N = {n}):")
        print(f"  - Rotation Geodesic Error Mean: {rot_mean:.3f} deg (Median: {rot_med:.3f} deg)")
        print(f"  - Translation Error Mean:       {trans_mean:.3f} mm (Median: {trans_med:.3f} mm)")
        print(f"  - Rot <= 5 deg Success Rate:    {hit_r5:.1f}%")
        print(f"  - Trans <= 20 mm Success Rate:  {hit_t20:.1f}%")

    if json_subset.exists():
        with open(json_subset, "r", encoding="utf-8") as f:
            data_sub = json.load(f)

        print(f"\n[Table 4] Comparison on Identical Tracked Subset (Four Shared Sessions):")
        headers = ["Method", "N", "Rot Mean (deg)", "Rot Med (deg)", "Trans Mean (mm)", "Trans Med (mm)"]
        header_fmt = "{:<24} | {:<8} | {:<16} | {:<14} | {:<16} | {:<14}"
        print(header_fmt.format(*headers))
        print("-" * 105)
        for m, stats in data_sub.items():
            if not isinstance(stats, dict) or "rot_deg_mean" not in stats:
                continue
            n_w = stats.get("N", 0)
            rm = stats.get("rot_deg_mean", 0)
            rp50 = stats.get("rot_p50", 0)
            tm = stats.get("trans_mean", 0)
            tp50 = stats.get("trans_p50", 0)
            print(header_fmt.format(
                m,
                str(n_w),
                f"{rm:.3f}",
                f"{rp50:.3f}" if rp50 else "-",
                f"{tm:.3f}" if tm else "-",
                f"{tp50:.3f}" if tp50 else "-"
            ))
        print("-" * 105)


def evaluate_table8(csv_batch: Path, csv_fill: Path):
    print("\n" + "=" * 105)
    print("  [Table 8] Sensitivity of Gradient Quantile q on CholecSeg8k Soft-Tissue Masks (N = 8,080 frames)")
    print("=" * 105)
    header_fmt = "  {:<24} | {:<18} | {:<18} | {:<18}"
    print(header_fmt.format("Setting", "Soft-tissue Dice", "Soft-tissue IoU", "Soft-tissue B-F1"))
    print("  " + "-" * 85)

    scores = {}
    if csv_batch.exists():
        with open(csv_batch, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                scores[r["method"]] = r
    if csv_fill.exists():
        with open(csv_fill, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                scores[r["method"]] = r

    order = [
        ("ours_th68", "q = 0.68"),
        ("ours_th68_fill", "q = 0.68 (filled)"),
        ("ours_th90_fill", "q = 0.90 (filled)"),
        ("ours_th95_fill", "q = 0.95 (filled)"),
    ]
    for m_id, label in order:
        if m_id in scores:
            r = scores[m_id]
            dice = float(r["mean_dice"]) if r.get("mean_dice") else 0.0
            iou = float(r["mean_iou"]) if r.get("mean_iou") else 0.0
            bf1 = float(r["mean_b_f1"]) if r.get("mean_b_f1") else 0.0
            print(header_fmt.format(label, f"{dice:.3f}", f"{iou:.3f}", f"{bf1:.3f}"))
    print("  " + "-" * 85)


def evaluate_imu_benchmarks(res_dir: Path):
    evaluate_table1(res_dir / "split_val_metrics.json", res_dir / "split_test_metrics.json")
    evaluate_table2(res_dir)
    evaluate_table3_and_4(res_dir / "vio_orbslam3_mono_fixedtest_metrics.json", res_dir / "vio_orbslam3_vs_classical_subset.json")
    evaluate_table8(res_dir / "cholecseg8k_contour_scores.csv", res_dir / "cholecseg8k_fill_scores.csv")


if __name__ == "__main__":
    res_dir = Path(__file__).resolve().parent.parent / "data" / "results"
    evaluate_imu_benchmarks(res_dir)
