#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional analyses of the 11 held-out sessions: session-averaged means and a per-session table, a shrinkage control
for the fused translation, an ungated gamma_R = 0 baseline, the per-window translation damage of the blends, the fused
translation with other GRU6D seeds and encoders, a causal inertial scale, the label-noise floor, Holm-adjusted p-values,
and time-offset diagnostics (per-session offsets, oracle per-session offset, and an offset estimate from consecutive
labeled frame pairs instead of 0.5 s windows). Writes review_addenda_results.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
import eval_fixedtest_aligned as ev  # noqa: E402
import eval_fusion_hybrid as fh  # noqa: E402
import eval_fusion_weights_validation as fw  # noqa: E402
import eval_gyro_time_alignment as ga  # noqa: E402

OUT = HERE / "review_addenda_results.json"
ENCODERS = ("ronin_resnet", "ronin_lstm", "tlio_resnet", "gru6d")
I3 = np.eye(3)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def macro(err, sess, names):
    per = [np.nanmean(err[sess == n]) for n in names]
    per = [x for x in per if not np.isnan(x)]
    return float(np.mean(per)), len(per)


def blend_t(recs, key, gR, gt, scales, tsrc, gate=True, tau=fh.TAU):
    """fh.blend with an arbitrary translation source tsrc(r) -> (3,)."""
    rot, tr, used = [], [], 0
    for r in recs:
        R, t = r["R_gyro"], tsrc(r)
        if key and r[key + "_ok"] and (not gate or ev.geo(r["R_" + key], R) <= tau):
            used += 1
            if gR > 0:
                R = ev.slerp(R, r["R_" + key], gR)
            t = gt * r["t_" + key] * scales[r["session"]] + (1 - gt) * t
        rot.append(ev.geo(R, r["R_gt"]))
        tr.append(float(np.linalg.norm(t - r["t_gt"])))
    return np.asarray(rot), np.asarray(tr), used


def scales_for(recs, key, tsrc):
    out = {}
    for s in dict.fromkeys(r["session"] for r in recs):
        rat = [np.linalg.norm(tsrc(r)) / np.linalg.norm(r["t_" + key]) for r in recs
               if r["session"] == s and r[key + "_ok"] and np.linalg.norm(r["t_" + key]) > 1e-9]
        out[s] = float(np.median(rat)) if rat else 1.0
    return out


def causal_blend(recs, key, gR, gt, warm=5):
    """Inertial scale from the median ratio over the PAST supported windows of the session (no look-ahead)."""
    rot, tr, used, past = [], [], 0, {}
    for r in recs:
        R, t = r["R_gyro"], r["gru"][0][1]
        hist = past.setdefault(r["session"], [])
        if r[key + "_ok"] and ev.geo(r["R_" + key], R) <= fh.TAU and len(hist) >= warm:
            used += 1
            if gR > 0:
                R = ev.slerp(R, r["R_" + key], gR)
            t = gt * r["t_" + key] * float(np.median(hist)) + (1 - gt) * t
        if r[key + "_ok"] and np.linalg.norm(r["t_" + key]) > 1e-9:
            hist.append(np.linalg.norm(r["gru"][0][1]) / np.linalg.norm(r["t_" + key]))
        rot.append(ev.geo(R, r["R_gt"]))
        tr.append(float(np.linalg.norm(t - r["t_gt"])))
    return np.asarray(rot), np.asarray(tr), used


def frame_pairs(sd, max_dt=0.3, max_rmse=2.0):
    rows = list(csv.DictReader(open(sd / "frames_labeled.csv", encoding="utf-8")))
    ok = [(float(r["frame_timestamp"]), ga.proj(np.array([float(r[f"R_{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)))
          for r in rows if int(float(r["pose_valid"])) == 1 and float(r["reproj_rmse"]) <= max_rmse]
    ts, te, A = [], [], []
    for (t0, R0), (t1, R1) in zip(ok[:-1], ok[1:]):
        if 0 < t1 - t0 <= max_dt:
            ts.append(t0)
            te.append(t1)
            A.append(R1 @ R0.T)
    return np.array(ts), np.array(te), np.array(A)


def main():
    al = json.loads((HERE / "gyro_time_alignment_results.json").read_text(encoding="utf-8"))
    g = al["variants"]["gyro_corrected_convention_offset"]
    delta, X = float(g["delta_s"]), np.asarray(g["R_CI"])
    hyb = json.loads((HERE / "fusion_hybrid_results.json").read_text(encoding="utf-8"))
    W = {k: (hyb["weights"][k]["gamma_R"], hyb["weights"][k]["gamma_t"]) for k in ("vo", "hf")}
    val = json.loads((HERE / "validation_sessions.json").read_text(encoding="utf-8"))["validation"]
    vrecs = fw.records(val, fw.gru6d_predict(val))
    fh.add_gyro(vrecs, delta, X)
    trecs, seeds = ev.build_records()
    sess = np.array([r["session"] for r in trecs])
    names = list(dict.fromkeys(sess.tolist()))
    vsess = np.array([r["session"] for r in vrecs])
    out = {"protocol": __doc__.strip().splitlines()[0], "N": len(trecs), "n_sessions": len(names)}
    gru = lambda r: r["gru"][0][1]  # noqa: E731
    sc = {k: ev.inertial_scales(trecs, k, 0) for k in ("vo", "hf", "slam")}

    # 1. Window share and per-session table ---------------------------------------------------------------------
    n_per = {n: int((sess == n).sum()) for n in names}
    top2 = sorted(n_per.values())[-2:]
    out["window_share"] = {"per_session": n_per, "largest_two_fraction": float(sum(top2) / len(trecs))}
    zr = np.array([ev.geo(I3, r["R_gt"]) for r in trecs])
    zt = np.array([np.linalg.norm(r["t_gt"]) for r in trecs])
    gy = np.array([ev.geo(r["R_gyro"], r["R_gt"]) for r in trecs])
    gr = np.array([ev.geo(r["gru"][0][0], r["R_gt"]) for r in trecs])
    it = np.array([np.linalg.norm(gru(r) - r["t_gt"]) for r in trecs])
    rows = {"inertial": fh.blend(trecs, None, 0, 0, None)}
    for k in ("vo", "hf"):
        rows[f"{k}_default"] = fh.blend(trecs, k, 0.5, 0.5, sc[k])
        rows[f"{k}_validation"] = fh.blend(trecs, k, *W[k], sc[k])
    rows["slam_default"] = fh.blend(trecs, "slam", 0.5, 0.5, sc["slam"], gate=False)
    vis = {k: ev.visual_only(trecs, k) for k in ("vo", "hf", "slam")}
    best_delta = {s["session"]: s["best_delta_s"] for s in al["per_session_offset"]["sessions"]}
    per = []
    for n in names:
        m = sess == n
        per.append({"session": n, "N": int(m.sum()), "best_delta_s": best_delta.get(n), "zero_rot": float(zr[m].mean()),
                    "gyro_rot": float(gy[m].mean()), "gru6d_rot": float(gr[m].mean()), "zero_trans": float(zt[m].mean()),
                    "inertial_trans": float(it[m].mean()),
                    "vo_only_rot": float(np.nanmean(vis["vo"][0][m])) if np.any(~np.isnan(vis["vo"][0][m])) else None,
                    "hf_only_rot": float(np.nanmean(vis["hf"][0][m])) if np.any(~np.isnan(vis["hf"][0][m])) else None,
                    "vo_validation_rot": float(rows["vo_validation"][0][m].mean()), "vo_validation_trans": float(rows["vo_validation"][1][m].mean()),
                    "hf_validation_rot": float(rows["hf_validation"][0][m].mean()), "hf_validation_trans": float(rows["hf_validation"][1][m].mean())})
    out["per_session"] = per

    # 2. Session-averaged (macro) means for the rows of the main table -----------------------------------------
    mac = {"zero_motion": (macro(zr, sess, names)[0], macro(zt, sess, names)[0]),
           "gyro_zeroT": (macro(gy, sess, names)[0], macro(zt, sess, names)[0]),
           "gru6d": (macro(gr, sess, names)[0], macro(it, sess, names)[0])}
    cl = np.load(HERE / "inertial_classical_aligned_errors.npz")
    if any("__" in k for k in cl.files):
        for kind in ("quat", "accel"):
            ks = [k for k in cl.files if k.startswith(kind) or kind in k]
            if ks:
                out.setdefault("classical_keys", []).extend(ks[:4])
    for k, (r, t, _) in rows.items():
        mac[k] = (macro(r, sess, names)[0], macro(t, sess, names)[0])
    for k, (r, t, _) in vis.items():
        mac[f"{k}_only"] = (macro(r, sess, names)[0], macro(t, sess, names)[0], macro(r, sess, names)[1])
    enc = {}
    for kind in ENCODERS:
        mr, mt, wr, wt = [], [], [], []
        for s in range(3):
            p = HERE / "inertial_baselines_aligned" / f"pred_errors_{kind}_s{s}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            rs = {k[: -len("__rot")]: z[k] for k in z.files if k.endswith("__rot")}
            tsd = {k[: -len("__trans")]: z[k] for k in z.files if k.endswith("__trans")}
            mr.append(np.mean([rs[n].mean() for n in rs]))
            mt.append(np.mean([tsd[n].mean() for n in tsd]))
            wr.append(np.concatenate(list(rs.values())).mean())
            wt.append(np.concatenate(list(tsd.values())).mean())
        enc[kind] = {"macro_rot": float(np.mean(mr)), "macro_trans": float(np.mean(mt)), "micro_rot": float(np.mean(wr)),
                     "micro_trans": float(np.mean(wt)), "n_seeds": len(mr)}
    out["macro_means"] = {k: {"rot": v[0], "trans": v[1]} for k, v in mac.items()}
    out["macro_means_encoders"] = enc

    # 3. Shrinkage control for the inertial translation -----------------------------------------------------------
    grid = np.round(np.arange(0.30, 1.0001, 0.01), 2)
    def shrink_err(recs, s):  # noqa: E306
        return float(np.mean([np.linalg.norm(s * gru(r) - r["t_gt"]) for r in recs]))
    v_curve = {float(s): shrink_err(vrecs, s) for s in grid}
    s_val = min(v_curve, key=v_curve.get)
    t_curve = {float(s): shrink_err(trecs, s) for s in grid}
    s_test = min(t_curve, key=t_curve.get)
    cos = np.array([np.dot(gru(r), r["t_gt"]) / (np.linalg.norm(gru(r)) * np.linalg.norm(r["t_gt"]) + 1e-12) for r in trecs])
    shr = {"s_validation": s_val, "test_trans_at_s_validation": t_curve[s_val], "s_test_oracle": s_test,
           "test_trans_at_s_test": t_curve[s_test], "test_trans_s1": t_curve[1.0],
           "macro_test_trans_at_s_validation": macro(np.array([np.linalg.norm(s_val * gru(r) - r["t_gt"]) for r in trecs]), sess, names)[0],
           "pred_norm_mean": float(np.mean([np.linalg.norm(gru(r)) for r in trecs])), "label_norm_mean": float(zt.mean()),
           "direction_cosine_mean": float(cos.mean()), "direction_cosine_median": float(np.median(cos)),
           "fraction_negative_cosine": float(np.mean(cos < 0))}
    shr["gain_vs_shrunk"] = {k: shr["test_trans_at_s_validation"] - float(rows[k][1].mean()) for k in ("vo_validation", "hf_validation", "vo_default", "hf_default", "slam_default")}
    shr["gain_vs_inertial"] = {k: float(rows["inertial"][1].mean() - rows[k][1].mean()) for k in shr["gain_vs_shrunk"]}
    out["shrinkage"] = shr

    # 4. Ungated gamma_R = 0 baseline with gamma_t selected on the validation sessions ------------------------------
    ung = {}
    for k in ("vo", "hf"):
        scv = ev.inertial_scales(vrecs, k, 0)
        curve = {float(gt): float(fh.blend(vrecs, k, 0.0, gt, scv, gate=False)[1].mean()) for gt in fh.GRID}
        gt_u = min(curve, key=curve.get)
        r, t, u = fh.blend(trecs, k, 0.0, gt_u, sc[k], gate=False)
        rg, tg_, ug = fh.blend(trecs, k, 0.0, W[k][1], sc[k], gate=True)
        ung[k] = {"gamma_t_validation_ungated": gt_u, "test_rot": float(r.mean()), "test_trans": float(t.mean()), "n_used": u,
                  "max_rot_excess": float((r - rows["inertial"][0]).max()),
                  "gated_gammaR0_test_rot": float(rg.mean()), "gated_gammaR0_test_trans": float(tg_.mean()), "gated_n_used": ug,
                  "validation_curve": curve}
    out["ungated_gammaR0"] = ung

    # 5. Translation damage of the blends (per-window increase over the inertial translation) ----------------------
    dmg = {}
    base_t = rows["inertial"][1]
    cfgs = {"vo_validation": ("vo", W["vo"], True), "hf_validation": ("hf", W["hf"], True), "vo_default": ("vo", (0.5, 0.5), True),
            "hf_default": ("hf", (0.5, 0.5), True), "vo_ungated_validation": ("vo", (0.0, ung["vo"]["gamma_t_validation_ungated"]), False),
            "hf_ungated_validation": ("hf", (0.0, ung["hf"]["gamma_t_validation_ungated"]), False), "slam_default": ("slam", (0.5, 0.5), False)}
    for name, (k, (gR, gt), gate) in cfgs.items():
        _, t, u = fh.blend(trecs, k, gR, gt, sc[k], gate=gate)
        d = t - base_t
        dmg[name] = {"n_used": u, "max_increase_mm": float(d.max()), "p95_increase_mm": float(np.percentile(d, 95)),
                     "p99_increase_mm": float(np.percentile(d, 99)), "n_increase_gt5mm": int((d > 5).sum()),
                     "n_increase_gt10mm": int((d > 10).sum()), "fraction_windows_worse": float(np.mean(d > 0)),
                     "mean_trans": float(t.mean())}
    out["translation_damage"] = dmg

    # 6. Other GRU6D seeds and other encoders as translation source (validation weights of GRU6D seed 0) ---------
    swap = {}
    for s in seeds:
        src = lambda r, s=s: r["gru"][s][1]  # noqa: E731
        row = {"inertial_trans": float(np.mean([np.linalg.norm(src(r) - r["t_gt"]) for r in trecs]))}
        for k in ("vo", "hf"):
            scs = scales_for(trecs, k, src)
            row[f"{k}_validation_trans"] = float(blend_t(trecs, k, *W[k], scs, src)[1].mean())
        swap[f"gru6d_seed{s}"] = row
    for kind in ENCODERS[:-1]:
        p = HERE / "inertial_baselines_aligned" / f"pred_{kind}_s0.npz"
        if not p.exists():
            continue
        z = np.load(p)
        idx = {}
        for r in trecs:
            k_ = idx.get(r["session"], 0)
            r["enc_" + kind] = z[r["session"]][k_, 9:12].astype(np.float64)
            idx[r["session"]] = k_ + 1
        src = lambda r, kind=kind: r["enc_" + kind]  # noqa: E731
        row = {"inertial_trans": float(np.mean([np.linalg.norm(src(r) - r["t_gt"]) for r in trecs]))}
        for k in ("vo", "hf"):
            scs = scales_for(trecs, k, src)
            row[f"{k}_validation_trans"] = float(blend_t(trecs, k, *W[k], scs, src)[1].mean())
        swap[f"{kind}_seed0"] = row
    out["translation_source_swap"] = swap

    # 7. Causal inertial scale ---------------------------------------------------------------------------------------
    out["causal_scale"] = {}
    for k in ("vo", "hf"):
        r, t, u = causal_blend(trecs, k, *W[k])
        out["causal_scale"][k] = {"rot": float(r.mean()), "trans": float(t.mean()), "n_used": u}

    # 8. Label-noise floor and PnP flips -------------------------------------------------------------------------------
    def floor(recs, tag):  # noqa: E306
        ag = np.array([ev.geo(r["R_gyro"], I3) for r in recs])
        alab = np.array([ev.geo(r["R_gt"], I3) for r in recs])
        st = ag < 0.3
        flip = (alab > 20) & (ag < 5)
        res = {"n_near_static": int(st.sum()), "label_angle_near_static_median": float(np.median(alab[st])) if st.any() else None,
               "label_angle_near_static_mean": float(alab[st].mean()) if st.any() else None,
               "label_angle_near_static_p95": float(np.percentile(alab[st], 95)) if st.any() else None,
               "n_flips_label_gt20_gyro_lt5": int(flip.sum()), "label_angle_max": float(alab.max()),
               "gyro_rot_mean_without_flips": float(np.mean([ev.geo(r["R_gyro"], r["R_gt"]) for r, f in zip(recs, flip) if not f])),
               "gru6d_rot_mean_without_flips": float(np.mean([ev.geo(r["gru"][0][0], r["R_gt"]) for r, f in zip(recs, flip) if not f])),
               "gyro_rot_mean": float(np.mean([ev.geo(r["R_gyro"], r["R_gt"]) for r in recs])),
               "gru6d_rot_mean": float(np.mean([ev.geo(r["gru"][0][0], r["R_gt"]) for r in recs])),
               "gyro_rot_median": float(np.median([ev.geo(r["R_gyro"], r["R_gt"]) for r in recs])),
               "gru6d_rot_median": float(np.median([ev.geo(r["gru"][0][0], r["R_gt"]) for r in recs]))}
        return res
    out["label_noise"] = {"test": floor(trecs, "test"), "validation": floor(vrecs, "validation")}
    out["label_noise"]["validation"]["gamma_R_curve_0_to_0.15"] = {k: v for k, v in hyb["weights"]["vo"]["validation_rot_by_gamma"].items() if float(k) <= 0.15}
    vflip = np.array([ev.geo(r["R_gt"], I3) > 20 and ev.geo(r["R_gyro"], I3) < 5 for r in vrecs])
    vkeep = [r for r, f in zip(vrecs, vflip) if not f]
    resel = {}
    for k in ("vo", "hf"):
        w_keep = fh.select(vkeep, k)
        scv = ev.inertial_scales(vrecs, k, 0)
        rot_med = {float(gg): float(np.median(fh.blend(vrecs, k, gg, 0.0, scv)[0])) for gg in fh.GRID}
        tr_med = {float(gg): float(np.median(fh.blend(vrecs, k, 0.0, gg, scv)[1])) for gg in fh.GRID}
        resel[k] = {"gamma_R_without_flips": w_keep["gamma_R"], "gamma_t_without_flips": w_keep["gamma_t"],
                    "closed_form_gamma_R_without_flips": w_keep["closed_form_gamma_R"],
                    "gamma_R_median_based": min(rot_med, key=rot_med.get), "gamma_t_median_based": min(tr_med, key=tr_med.get)}
    out["weight_reselection"] = resel

    # 8b. Session-averaged means of the classical baselines and per-session PnP residuals -------------------------------
    cls = np.load(HERE / "inertial_classical_aligned_errors.npz")
    csess = cls["session"].astype(str)
    out["macro_means_classical"] = {k: float(np.mean([cls[k][csess == n].mean() for n in names])) for k in ("rot_gyro", "rot_quat", "trans_zero", "trans_ins")}
    for p in per:
        meta = list(csv.DictReader(open(fw.SESS / p["session"] / ev.DS / "meta.csv", encoding="utf-8")))
        rr = np.array([float(m["reproj_rmse_end"]) for m in meta] + [float(m["reproj_rmse_start"]) for m in meta])
        p["reproj_rmse_median_px"] = float(np.median(rr))
        p["quat_rot"] = float(cls["rot_quat"][csess == p["session"]].mean())

    # 8c. Full-statistics rows for the main table ------------------------------------------------------------------------
    trows = {}
    t_shr = np.array([np.linalg.norm(s_val * gru(r) - r["t_gt"]) for r in trecs])
    trows["inertial_shrunk"] = fh.summ(gy, t_shr)
    for k in ("vo", "hf"):
        gt_u = ung[k]["gamma_t_validation_ungated"]
        r, t, u = fh.blend(trecs, k, 0.0, gt_u, sc[k], gate=False)
        trows[f"{k}_ungated_gammaR0"] = fh.summ(r, t, u)
        trows[f"{k}_ungated_gammaR0"]["gamma_t"] = gt_u
    out["table_rows"] = trows

    # 9. Holm-adjusted p-values ------------------------------------------------------------------------------------------
    summ = json.loads((HERE / "inertial_baselines_aligned" / "inertial_baselines_summary.json").read_text(encoding="utf-8"))
    p4 = {"inertial_vs_orbvo_rot": hyb["stats"]["inertial_vs_orbvo_rot"]["p"], "gru6d_vs_gyro_rot": summ["gru6d_vs_gyro"]["rot_p"]}
    for k in ("vo_default", "vo_validation", "hf_default", "hf_validation", "slam_default"):
        p4[k + "_rot"] = hyb["stats"][k + "_rot"]["p"]
        p4[k + "_trans"] = hyb["stats"][k + "_trans"]["p"]
    for kind in ENCODERS[:-1]:
        p4[kind + "_rot"] = summ[kind]["vs_retrained"]["rot_p"]
        p4[kind + "_trans"] = summ[kind]["vs_retrained"]["trans_p"]
    keys = list(p4)
    adj = holm([p4[k] for k in keys])
    out["holm_table_session_stats"] = {k: {"p": p4[k], "p_holm": a} for k, a in zip(keys, adj)}
    t9 = {"A_real18": {"VGGT": 0.87, "EndoDAC": 0.73, "CUT3R": 0.26, "Reloc3r": 0.13, "AF-SfMLearner": 0.15, "Endo-FASt3r": 0.009, "Pi3": 0.048, "Static": 7.6e-6},
          "B_sim71": {"DROID-SLAM": 5.2e-7, "VGGT": 0.22, "CUT3R": 0.027, "EndoDAC": 3.7e-4, "AF-SfMLearner": 1.1e-6, "Reloc3r": 4.9e-7,
                      "Endo-FASt3r": 2.2e-10, "Pi3": 6.5e-8, "Static": 2.4e-13},
          "C_c3vd10": {"AF-SfMLearner": 0.014, "DROID-SLAM": 0.064, "EndoDAC": 0.027, "Endo-FASt3r": 0.006, "Pi3": 0.010, "Reloc3r": 0.11, "Static": 0.002},
          "pure_rotation21": {"DROID-SLAM": 9.5e-7, "VGGT": 9.5e-7, "CUT3R": 0.11, "EndoDAC": 0.27, "AF-SfMLearner": 0.14, "Reloc3r": 0.001,
                              "Endo-FASt3r": 9.5e-7, "Pi3": 9.5e-7, "Static": 0.018}}
    out["holm_visual_tables"] = {blk: dict(zip(d.keys(), holm(list(d.values())))) for blk, d in t9.items()}

    # 10. Time-offset diagnostics -------------------------------------------------------------------------------------------
    tr_s, te_s = ga.sessions()
    te = ga.load(te_s)
    oracle_rows, pooled_err = [], []
    for s in te:
        d_s = best_delta[s["name"]]
        e_fixed = ga.errors(s["A"], np.transpose(ga.body_increment(s["gyro"], s["ts"], s["te"], delta), (0, 2, 1)), X)
        e_or = ga.errors(s["A"], np.transpose(ga.body_increment(s["gyro"], s["ts"], s["te"], float(d_s)), (0, 2, 1)), X)
        oracle_rows.append({"session": s["name"], "n": int(len(s["A"])), "best_delta_s": d_s, "gyro_rot_fixed_delta": float(e_fixed.mean()),
                            "gyro_rot_session_delta": float(e_or.mean())})
        pooled_err.append(e_or)
    pooled = np.concatenate(pooled_err)
    test_deltas = [r["best_delta_s"] for r in oracle_rows]
    out["time_offset"] = {"test_sessions": oracle_rows, "oracle_session_delta_pooled_mean": float(pooled.mean()),
                          "oracle_session_delta_macro_mean": float(np.mean([r["gyro_rot_session_delta"] for r in oracle_rows])),
                          "fixed_delta_macro_mean": float(np.mean([r["gyro_rot_fixed_delta"] for r in oracle_rows])),
                          "test_delta_median": float(np.median(test_deltas)), "test_delta_min": float(min(test_deltas)), "test_delta_max": float(max(test_deltas)),
                          "n_sessions_delta_within_0.05_of_minus0.50": int(sum(abs(d + 0.50) <= 0.05 for d in test_deltas)),
                          "calibration_sessions_outside_0.10": [(s["session"], s["n"], s["best_delta_s"]) for s in al["per_session_offset"]["sessions"]
                                                                 if abs(s["best_delta_s"] + 0.50) > 0.10]}
    # Offset from consecutive labeled frame pairs (about 0.17 s apart) on the 49 calibration sessions, fixed R_CI.
    tracks = {s["name"]: s["gyro"] for s in ga.load(tr_s)}
    pairs = [(n, *frame_pairs(IMUCAM_ROOT / "Data_IMU_Camera_Pose_Sequences" / n)) for n in tracks]
    A_all = np.concatenate([p[3] for p in pairs])
    scan = []
    for d in ga.DELTAS:
        B = np.concatenate([np.transpose(ga.body_increment(tracks[n], ts, te_, float(d)), (0, 2, 1)) for n, ts, te_, _ in pairs])
        scan.append((float(d), float(ga.errors(A_all, B, X).mean())))
    d_pair = min(scan, key=lambda x: x[1])
    dts = np.concatenate([te_ - ts for _, ts, te_, _ in pairs])
    out["time_offset"]["consecutive_frame_pairs"] = {"n_pairs": int(len(A_all)), "mean_dt_s": float(dts.mean()), "best_delta_s": d_pair[0],
                                                      "residual_at_best": d_pair[1], "residual_at_zero": dict(scan)[0.0],
                                                      "residual_at_minus0.50": dict(scan)[-0.5]}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("window share of the two largest sessions:", round(out["window_share"]["largest_two_fraction"], 3))
    print("PER-SESSION")
    for p in per:
        print(f"  {p['session'][-6:]} N {p['N']:4d} delta {p['best_delta_s']:+.2f} zero {p['zero_rot']:.2f} gyro {p['gyro_rot']:.2f} gru {p['gru6d_rot']:.2f} "
              f"| trans zero {p['zero_trans']:.2f} inert {p['inertial_trans']:.2f} vo {p['vo_validation_trans']:.2f} hf {p['hf_validation_trans']:.2f}")
    print("MACRO", {k: (round(v['rot'], 3), round(v['trans'], 3)) for k, v in out["macro_means"].items()})
    print("MACRO ENC", {k: (round(v['macro_rot'], 3), round(v['macro_trans'], 3)) for k, v in enc.items()})
    print("SHRINK", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in shr.items() if not isinstance(v, dict)})
    print("SHRINK gains vs shrunk", {k: round(v, 3) for k, v in shr["gain_vs_shrunk"].items()})
    print("UNGATED", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "validation_curve"} for k, v in ung.items()})
    print("DAMAGE", {k: (v["n_used"], round(v["max_increase_mm"], 1), round(v["p95_increase_mm"], 2), v["n_increase_gt5mm"], round(v["fraction_windows_worse"], 3)) for k, v in dmg.items()})
    print("SWAP", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in swap.items()})
    print("CAUSAL", out["causal_scale"])
    print("LABEL NOISE", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in out["label_noise"].items()})
    print("WEIGHT RESELECTION", resel)
    print("MACRO CLASSICAL", {k: round(v, 3) for k, v in out["macro_means_classical"].items()})
    print("TABLE ROWS", {k: (round(v["rot_mean"], 3), round(v["trans_mean"], 3), round(v["trans_p50"], 3), round(v["trans_p95"], 3), round(v["hit_trans20"], 3)) for k, v in trows.items()})
    print("HOLM session stats", {k: (round(v['p'], 4), round(v['p_holm'], 3)) for k, v in out["holm_table_session_stats"].items()})
    print("HOLM visual", {b: {k: round(v, 3) for k, v in d.items()} for b, d in out["holm_visual_tables"].items()})
    print("TIME OFFSET", {k: v for k, v in out["time_offset"].items() if k != "test_sessions"})


if __name__ == "__main__":
    main()
