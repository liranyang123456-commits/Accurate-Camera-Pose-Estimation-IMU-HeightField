#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CholecSeg8k-Hard stratification for Table 8 (R1-5).

Reuses the already-computed per-frame Dice/IoU/B-F1 from:
  D:\\reloc3r\\eval_contour_CC_outputs_batch\\archive_soft_tissue\\summary.csv
  D:\\reloc3r\\eval_contour_CC_outputs_filled_from_contour\\archive_soft_tissue\\summary.csv

Cue definitions are registered *before* looking at Dice:
  blood     : watershed class 7 (gray 24) pixel fraction >= 0.005
  specular  : top 10% of (V>=0.92 and S<=0.20) pixel fraction
  blur      : bottom 10% of Laplacian variance on grayscale

Quantile cutoffs are computed on the 8080-frame cue distributions, not on metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ARCHIVE = Path(r"D:\360极速浏览器X下载\archive")
BATCH_SUMMARY = Path(r"D:\reloc3r\eval_contour_CC_outputs_batch\archive_soft_tissue\summary.csv")
FILLED_SUMMARY = Path(
    r"D:\reloc3r\eval_contour_CC_outputs_filled_from_contour\archive_soft_tissue\summary.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "cholecseg8k_hard"

WATERSHED_BLOOD_GRAY = 24  # class 7 Blood #242424
BLOOD_FRAC_MIN = 0.005
SPECULAR_V_MIN = 235  # 0.92 * 255
SPECULAR_S_MAX = 51  # 0.20 * 255
SPECULAR_Q = 0.90
BLUR_Q = 0.10

METHODS = [
    "ours_th68",
    "ours_th68_fill",
    "ours_th90_fill",
    "ours_th95_fill",
]


def _imread(path: Path, flags: int) -> np.ndarray | None:
    sp = str(path)
    try:
        sp.encode("ascii")
        img = cv2.imread(sp, flags)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        data = np.fromfile(sp, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _list_endo(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*_endo.png") if p.is_file()]
    paths.sort(key=lambda x: str(x).replace("\\", "/").lower())
    return paths


def _cues_one(img_path: Path, archive: Path) -> dict:
    rel = img_path.relative_to(archive).as_posix()
    ws_path = img_path.with_name(img_path.stem + "_watershed_mask.png")
    bgr = _imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {img_path}")
    h, w = bgr.shape[:2]
    n = float(h * w)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    specular = (hsv[:, :, 2] >= SPECULAR_V_MIN) & (hsv[:, :, 1] <= SPECULAR_S_MAX)
    specular_frac = float(specular.mean())

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_v = float(hsv[:, :, 2].mean())

    blood_frac = 0.0
    ws_ok = False
    if ws_path.is_file():
        ws = _imread(ws_path, cv2.IMREAD_GRAYSCALE)
        if ws is not None:
            ws_ok = True
            blood_frac = float((ws == WATERSHED_BLOOD_GRAY).mean())

    return {
        "rel_path": rel,
        "h": int(h),
        "w": int(w),
        "n_px": int(n),
        "blood_frac": blood_frac,
        "specular_frac": specular_frac,
        "lap_var": lap_var,
        "mean_v": mean_v,
        "ws_ok": bool(ws_ok),
    }


def _load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["rel_path", "method", "dice", "iou", "b_f1"])
    df["rel_path"] = df["rel_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub_m = df[df["method"] == method]
        if sub_m.empty:
            continue
        for stratum, sub in sub_m.groupby("stratum", sort=False):
            rows.append(
                {
                    "method": method,
                    "stratum": stratum,
                    "n": int(len(sub)),
                    "n_frames": int(sub["rel_path"].nunique()),
                    "mean_dice": float(sub["dice"].mean()),
                    "mean_iou": float(sub["iou"].mean()),
                    "mean_b_f1": float(sub["b_f1"].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_dir():
        raise SystemExit(f"archive not found: {ARCHIVE}")
    images = _list_endo(ARCHIVE)
    if len(images) != 8080:
        print(f"WARNING: expected 8080 endo frames, found {len(images)}")

    cue_path = OUT_DIR / "per_frame_cues.csv"
    if cue_path.is_file():
        cues = pd.read_csv(cue_path)
        print(f"loaded cached cues: {cue_path} n={len(cues)}")
    else:
        rows = []
        for i, p in enumerate(images, 1):
            rows.append(_cues_one(p, ARCHIVE))
            if i % 400 == 0 or i == len(images):
                print(f"cues {i}/{len(images)}")
        cues = pd.DataFrame(rows)
        cues.to_csv(cue_path, index=False)

    if int(cues["ws_ok"].sum()) != len(cues):
        print(f"WARNING: watershed missing on {int((~cues['ws_ok']).sum())} frames")

    spec_cut = float(cues["specular_frac"].quantile(SPECULAR_Q))
    blur_cut = float(cues["lap_var"].quantile(BLUR_Q))
    protocol = {
        "n_frames": int(len(cues)),
        "archive": str(ARCHIVE),
        "blood_rule": f"blood_frac >= {BLOOD_FRAC_MIN} (watershed class 7 gray={WATERSHED_BLOOD_GRAY})",
        "specular_rule": (
            f"specular_frac >= P{int(SPECULAR_Q*100)} "
            f"(V>={SPECULAR_V_MIN}/255 and S<={SPECULAR_S_MAX}/255); cutoff={spec_cut:.6g}"
        ),
        "blur_rule": f"lap_var <= P{int(BLUR_Q*100)}; cutoff={blur_cut:.6g}",
        "cue_percentiles": {
            "blood_frac": cues["blood_frac"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict(),
            "specular_frac": cues["specular_frac"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict(),
            "lap_var": cues["lap_var"].quantile([0.01, 0.05, 0.1, 0.5, 0.9]).to_dict(),
        },
        "cutoffs": {
            "blood_frac_min": BLOOD_FRAC_MIN,
            "specular_frac_min": spec_cut,
            "lap_var_max": blur_cut,
        },
    }
    cues["is_blood"] = cues["blood_frac"] >= BLOOD_FRAC_MIN
    cues["is_specular"] = cues["specular_frac"] >= spec_cut
    cues["is_blur"] = cues["lap_var"] <= blur_cut
    cues["is_hard"] = cues["is_blood"] | cues["is_specular"] | cues["is_blur"]
    cues["is_clean"] = ~cues["is_hard"]

    counts = {
        "all": int(len(cues)),
        "blood": int(cues["is_blood"].sum()),
        "specular": int(cues["is_specular"].sum()),
        "blur": int(cues["is_blur"].sum()),
        "hard_union": int(cues["is_hard"].sum()),
        "clean": int(cues["is_clean"].sum()),
        "blood_and_specular": int((cues["is_blood"] & cues["is_specular"]).sum()),
        "blood_and_blur": int((cues["is_blood"] & cues["is_blur"]).sum()),
        "specular_and_blur": int((cues["is_specular"] & cues["is_blur"]).sum()),
    }
    protocol["subset_counts"] = counts
    (OUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cues.to_csv(OUT_DIR / "per_frame_tags.csv", index=False)
    print("subset counts:", counts)
    print("cutoffs:", protocol["cutoffs"])

    scores = pd.concat(
        [_load_scores(BATCH_SUMMARY), _load_scores(FILLED_SUMMARY)],
        ignore_index=True,
    )
    scores = scores[scores["method"].isin(METHODS)].copy()
    tags = cues[
        ["rel_path", "is_blood", "is_specular", "is_blur", "is_hard", "is_clean"]
    ].copy()
    merged = scores.merge(tags, on="rel_path", how="left")
    if merged[["is_hard"]].isna().any(axis=None):
        n_miss = int(merged["is_hard"].isna().sum())
        print(f"WARNING: {n_miss} score rows failed to join tags")
        merged = merged.dropna(subset=["is_hard"])

    parts = [merged.assign(stratum="all")]
    flag_to_name = {
        "is_blood": "blood",
        "is_specular": "specular",
        "is_blur": "blur",
        "is_hard": "hard_union",
        "is_clean": "clean",
    }
    for col, name in flag_to_name.items():
        parts.append(merged.loc[merged[col].astype(bool)].assign(stratum=name))
    long = pd.concat(parts, ignore_index=True)
    table = _agg(long)
    table_path = OUT_DIR / "stratified_metrics.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Keep a compact paper table: 4 methods x {all, blood, specular, blur, hard, clean}
    wide_rows = []
    for method in METHODS:
        row = {"method": method}
        for stratum in ["all", "blood", "specular", "blur", "hard_union", "clean"]:
            sub = table[(table["method"] == method) & (table["stratum"] == stratum)]
            if sub.empty:
                continue
            row[f"{stratum}_n"] = int(sub["n_frames"].iloc[0])
            row[f"{stratum}_dice"] = float(sub["mean_dice"].iloc[0])
            row[f"{stratum}_iou"] = float(sub["mean_iou"].iloc[0])
            row[f"{stratum}_bf1"] = float(sub["mean_b_f1"].iloc[0])
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(OUT_DIR / "table8_hard_wide.csv", index=False)

    # Example frames for qualitative figure (do not use Dice to pick).
    examples = []
    for name, col, how in [
        ("blood", "blood_frac", False),
        ("specular", "specular_frac", False),
        ("blur", "lap_var", True),
    ]:
        sub = cues.sort_values(col, ascending=how).head(6)
        for rank, rec in enumerate(sub.itertuples(index=False), 1):
            examples.append(
                {
                    "kind": name,
                    "rank": rank,
                    "rel_path": rec.rel_path,
                    "blood_frac": rec.blood_frac,
                    "specular_frac": rec.specular_frac,
                    "lap_var": rec.lap_var,
                }
            )
    pd.DataFrame(examples).to_csv(OUT_DIR / "example_frames.csv", index=False)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
