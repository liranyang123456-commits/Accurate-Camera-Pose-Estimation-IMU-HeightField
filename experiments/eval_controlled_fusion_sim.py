#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled fusion study with simulated inertial increments on the simulated sequences and the real segments.

The visual increments are those of the height-field tracker (hf_benchmarks/<bench>/heightfield, the 64 evaluated
frames). Inertial increments are synthesized from the reference increments: rotation dR_gt Exp(n_R) with
n_R ~ N(0, sigma_R^2 I) per axis, translation dt_gt + n_t with n_t ~ N(0, (0.2 |dt_gt| + 0.1 mm)^2 I).
Fusion follows Eqs. (vi_gate)-(vi_blend) of the paper: if the geodesic disagreement of the two rotations is at most
tau, R = slerp(R_imu, R_vis, gamma) and t = gamma s t_vis + (1 - gamma) t_imu, with s the per-sequence median ratio of
inertial to visual increment norms; otherwise the inertial increment is used. gamma* = var_imu / (var_imu + var_vis)
is evaluated with the visual rotation-error variance of all increments ("ungated") and of the increments that pass
the gate ("gated"); both use the reference and are oracle choices. Five noise seeds per configuration.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
from paths import ENDOSIM_ROOT  # noqa: E402
sys.path.insert(0, str(ENDOSIM_ROOT))
from endosim.eval.metrics import load_pose_txt  # noqa: E402
from endosim.eval.protocol import list_color_frames, select_frame_indices  # noqa: E402
from paths import eval_with_scale_correction  # noqa: E402

LISTS = ENDOSIM_ROOT / "lists"
BENCH = {"sim92": "simtest92.txt", "real18": "egomo_gtraj.txt"}
SIGMA_R = (0.1, 0.5, 1.0, 2.0)
GAMMA_GRID = tuple(round(0.1 * k, 1) for k in range(1, 11))
TAU0 = 3.0
SEEDS = range(5)
T_REL, T_ABS = 0.2, 0.1


def geo(R1, R2):
    return float(np.degrees(Rotation.from_matrix(R1.T @ R2).magnitude()))


def increments(T):
    return [np.linalg.inv(T[i - 1]) @ T[i] for i in range(1, len(T))]


def load_bench(name):
    seqs = [ln.strip() for ln in open(LISTS / BENCH[name], encoding="utf-8") if ln.strip()]
    motion = json.loads((HERE / "hf_benchmarks" / f"{name}_reeval.json").read_text(encoding="utf-8")).get("motion", {})
    out = []
    for sdir in seqs:
        sid = os.path.basename(sdir.rstrip("/\\"))
        gt_all = load_pose_txt(os.path.join(sdir, "pose_c2w.txt"))
        n = min(len(list_color_frames(sdir)), len(gt_all))
        gt = gt_all[:n][select_frame_indices(gt_all[:n], "uniform", max_frames=64)]
        hf = load_pose_txt(str(HERE / "hf_benchmarks" / name / "heightfield" / f"{sid}_est_c2w.txt"))
        m = min(len(gt), len(hf))
        dgt, dvis = increments(gt[:m]), increments(hf[:m])
        out.append({"sid": sid, "gt": gt[:m], "dgt": dgt, "dvis": dvis, "orbital": motion.get(sid) == "orbital",
                    "evis": np.array([geo(v[:3, :3], g[:3, :3]) for v, g in zip(dvis, dgt)])})
    return out


def simulate(dgt, sigma_r, rng):
    res = []
    for d in dgt:
        R = d[:3, :3] @ Rotation.from_rotvec(rng.normal(0.0, np.radians(sigma_r), 3)).as_matrix()
        t = d[:3, 3] + rng.normal(0.0, T_REL * np.linalg.norm(d[:3, 3]) + T_ABS, 3)
        res.append((R, t))
    return res


def scale(dvis, dimu):
    r = [np.linalg.norm(ti) / np.linalg.norm(v[:3, 3]) for v, (_, ti) in zip(dvis, dimu)
         if np.linalg.norm(v[:3, 3]) > 1e-9 and np.linalg.norm(ti) > 1e-9]
    return float(np.median(r)) if r else 1.0


def fuse(dvis, dimu, gamma, tau, rot_only=False):
    s = scale(dvis, dimu)
    out, acc, dis = [], [], []
    for v, (Ri, ti) in zip(dvis, dimu):
        Rv, tv = v[:3, :3], v[:3, 3] * s
        d = geo(Rv, Ri)
        if gamma > 0 and d <= tau:
            acc.append(True)
            R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ri, Rv])))([gamma]).as_matrix()[0]
            t = ti if rot_only else gamma * tv + (1.0 - gamma) * ti
        else:
            acc.append(False)
            R, t = Ri, ti
        out.append((R, t))
        dis.append(d)
    return out, np.array(acc), np.array(dis)


def chain(inc):
    T = [np.eye(4)]
    for R, t in inc:
        d = np.eye(4)
        d[:3, :3], d[:3, 3] = R, t
        T.append(T[-1] @ d)
    return np.stack(T)


def evaluate(data, sims, key, gamma, tau, rot_only=False):
    ra, ro, rt, at, viol, exc, na, ntot = [], [], [], [], [], [], 0, 0
    for s, dimu in zip(data, sims):
        if key == "imu":
            inc, acc, dis = dimu, np.zeros(len(dimu), bool), np.zeros(len(dimu))
        else:
            inc, acc, dis = fuse(s["dvis"], dimu, gamma, tau, rot_only)
        e_f = np.array([geo(R, d[:3, :3]) for (R, _), d in zip(inc, s["dgt"])])
        e_i = np.array([geo(R, d[:3, :3]) for (R, _), d in zip(dimu, s["dgt"])])
        ra.append(e_f)
        (ro if s["orbital"] else rt).append(e_f)
        exc.append(e_f - e_i)
        if np.isfinite(tau):
            viol.append(e_f - e_i - gamma * tau)
        na, ntot = na + int(acc.sum()), ntot + len(inc)
        if not s["orbital"]:
            at.append(float(eval_with_scale_correction(chain(inc), s["gt"])["ate_sim3"]["rmse"]))
    ra = np.concatenate(ra)
    return {"rot_all": float(ra.mean()), "rot_orb": float(np.concatenate(ro).mean()) if ro else None,
            "rot_trans": float(np.concatenate(rt).mean()), "ate_trans": float(np.mean(at)),
            "max_excess": float(np.concatenate(exc).max()),
            "max_excess_minus_bound": float(np.concatenate(viol).max()) if viol else None,
            "accepted": na / ntot}


def run(name):
    data = load_bench(name)
    evis = np.concatenate([s["evis"] for s in data])
    var_vis = float(np.mean(np.radians(evis) ** 2))
    res = {}
    for sr in SIGMA_R:
        var_imu = 3.0 * np.radians(sr) ** 2
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * seed + int(10 * sr))
            sims = [simulate(s["dgt"], sr, rng) for s in data]
            acc3 = np.concatenate([fuse(s["dvis"], dimu, 1.0, TAU0)[1] for s, dimu in zip(data, sims)])
            var_vis_g = float(np.mean(np.radians(evis[acc3]) ** 2)) if acc3.any() else var_vis
            g_un = var_imu / (var_imu + var_vis)
            g_g = var_imu / (var_imu + var_vis_g)
            r = {"gamma_star_ungated": g_un, "gamma_star_gated": g_g, "imu": evaluate(data, sims, "imu", 0.0, np.inf)}
            for g in GAMMA_GRID:
                r[f"g{g}_t{TAU0}"] = evaluate(data, sims, "f", g, TAU0)
            for tau in (1.0, 5.0, np.inf):
                r[f"g0.5_t{tau}"] = evaluate(data, sims, "f", 0.5, tau)
            r["gstar_ungated"] = evaluate(data, sims, "f", g_un, TAU0)
            r["gstar_gated"] = evaluate(data, sims, "f", g_g, TAU0)
            r["g0.5_rot_only"] = evaluate(data, sims, "f", 0.5, TAU0, rot_only=True)
            per_seed.append(r)
        agg = {}
        for k in per_seed[0]:
            if isinstance(per_seed[0][k], dict):
                agg[k] = {m: ([float(np.mean([p[k][m] for p in per_seed])), float(np.std([p[k][m] for p in per_seed]))]
                              if per_seed[0][k][m] is not None else None) for m in per_seed[0][k]}
            else:
                agg[k] = [float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed]))]
        res[str(sr)] = agg
        best = min(GAMMA_GRID, key=lambda g: agg[f"g{g}_t{TAU0}"]["rot_all"][0])
        print(f"{name} sigma_R={sr}: imu {agg['imu']['rot_all'][0]:.3f} | g0.5 {agg['g0.5_t3.0']['rot_all'][0]:.3f} | "
              f"no gate {agg['g0.5_tinf']['rot_all'][0]:.3f} | g*un {agg['gamma_star_ungated'][0]:.3f}->{agg['gstar_ungated']['rot_all'][0]:.3f} | "
              f"g*g {agg['gamma_star_gated'][0]:.3f}->{agg['gstar_gated']['rot_all'][0]:.3f} | best grid g={best} "
              f"{agg[f'g{best}_t{TAU0}']['rot_all'][0]:.3f} | excess-bound max {agg['g0.5_t3.0']['max_excess_minus_bound'][0]:.2e}", flush=True)
    return {"n_sequences": len(data), "n_increments": int(len(evis)), "n_orbital": int(sum(s["orbital"] for s in data)),
            "visual_only": {"rot_all": float(evis.mean()), "rmse_deg": float(np.degrees(np.sqrt(var_vis))),
                            "rot_orb": float(np.concatenate([s["evis"] for s in data if s["orbital"]]).mean())
                            if any(s["orbital"] for s in data) else None},
            "results": res}


def main():
    out = {"doc": __doc__.strip().splitlines()[0], "sigma_t": {"relative": T_REL, "absolute_mm": T_ABS},
           "seeds": len(SEEDS), "tau_deg": TAU0, "gamma_grid": GAMMA_GRID}
    for name in BENCH:
        out[name] = run(name)
    (HERE / "controlled_fusion_sim_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
