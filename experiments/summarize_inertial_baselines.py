#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mean +/- s.d. over seeds and session-level comparisons of the learned inertial encoders.

Session-level comparisons use GRU6D re-trained in the same loop (seed-averaged) as the reference
when all of its seeds are available, and the evaluated GRU6D checkpoint otherwise.
Gain = error(reference) - error(encoder); a positive value favors the encoder.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
D = HERE / "inertial_baselines"
KINDS = [("ronin_resnet", "RoNIN-ResNet \\citep{herath2020ronin}"), ("ronin_lstm", "RoNIN-LSTM \\citep{herath2020ronin}"),
         ("tlio_resnet", "TLIO-ResNet \\citep{liu2020tlio}"), ("gru6d", "GRU6D (re-trained, same loop)")]
SEEDS = (0, 1, 2)


def load_npz(p):
    z = np.load(p)
    ses = sorted({k.split("__")[0] for k in z.files})
    return ses, {s: (z[f"{s}__rot"], z[f"{s}__trans"]) for s in ses}


def seed_avg(kind, ses):
    per = {s: [] for s in ses}
    pert = {s: [] for s in ses}
    for sd in SEEDS:
        p = D / f"pred_errors_{kind}_s{sd}.npz"
        if p.exists():
            _, e = load_npz(p)
            for s in ses:
                per[s].append(e[s][0])
                pert[s].append(e[s][1])
    if not per[ses[0]]:
        return None, 0
    return {s: (np.mean(per[s], axis=0), np.mean(pert[s], axis=0)) for s in ses}, len(per[ses[0]])


def compare(ref, enc, ses, seed=7):
    out = {}
    rng = np.random.default_rng(seed)
    for k, name in ((0, "rot"), (1, "trans")):
        groups = [ref[s][k] - enc[s][k] for s in ses]
        g_ses = np.array([g.mean() for g in groups])
        boot = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(5000)]
        out[f"{name}_gain"] = float(np.concatenate(groups).mean())
        out[f"{name}_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out[f"{name}_sessions_better"] = int((g_ses > 0).sum())
        out[f"{name}_p"] = float(stats.wilcoxon(g_ses).pvalue)
    return out


def fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def main():
    ses, ckpt = load_npz(D / "pred_errors_reference_gru6d.npz")
    res = {}
    for f in sorted(D.glob("inertial_baselines_results_s*.json")):
        res.update(json.loads(f.read_text(encoding="utf-8"))["runs"])
    retrained, n_re = seed_avg("gru6d", ses)
    use_retrained = retrained is not None and n_re == len(SEEDS)
    ref = retrained if use_retrained else ckpt
    ref_name = "GRU6D (re-trained)" if use_retrained else "GRU6D"
    rows, srows, out = [], [], {"session_reference": ref_name}
    for kind, label in KINDS:
        runs = [r for r in (res.get(f"{kind}_s{s}") for s in SEEDS) if r]
        if not runs:
            continue

        def m(key, nd=3):
            return f"{np.mean([r[key] for r in runs]):.{nd}f}"

        def ms(key, nd=3):
            v = np.array([r[key] for r in runs])
            return m(key, nd) + (f"$\\pm${v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")
        rows.append(f"{label} & {ms('rot_mean')} / {m('rot_p50')} / {m('rot_p95')} & {ms('trans_mean')} / {m('trans_p50')} / {m('trans_p95')} & "
                    f"{m('hit_rot5')} / {m('hit_trans20')} & 2312 \\\\")
        enc, _ = seed_avg(kind, ses)
        o = {"n_seeds": len(runs), "params_M": round(runs[0]["params"] / 1e6, 2),
             "rot_mean": [r["rot_mean"] for r in runs], "trans_mean": [r["trans_mean"] for r in runs],
             "vs_checkpoint": compare(ckpt, enc, ses)}
        if use_retrained and kind != "gru6d":
            o["vs_retrained"] = compare(retrained, enc, ses)
        out[kind] = o
        if kind == "gru6d" and use_retrained:
            continue
        c = o["vs_retrained"] if use_retrained else o["vs_checkpoint"]
        name = label.split(" \\citep")[0]
        for key, metric in (("rot", "rot.\\ ($^\\circ$)"), ("trans", "trans.\\ (mm)")):
            srows.append(f"{name} vs.\\ {ref_name} & {metric} & ${c[key + '_gain']:.3f}$ & "
                         f"$[{c[key + '_ci'][0]:.3f},\\,{c[key + '_ci'][1]:.3f}]$ & ${c[key + '_sessions_better']}/11$ & "
                         f"${fmt_p(c[key + '_p'])}$ \\\\")
    (HERE / "rows_inertial_baselines.tex").write_text("\n".join(rows), encoding="utf-8")
    (HERE / "rows_inertial_session.tex").write_text("\n".join(srows), encoding="utf-8")
    (D / "inertial_baselines_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(rows))
    print("\n".join(srows))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
