#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the inertial regressor on the MITI intervention (6-channel windows of build_miti_windows.py) with the recipe of
the main paper (train_inertial_baselines.train_one: AdamW 1e-3, batch 64, 8 x geodesic + smooth-L1, plateau decay,
best-validation checkpoint). Input and pose scalers are fitted on the MITI training split. Encoders: GRU6D (input_dim 6)
and, as controls, RoNIN-LSTM and TLIO-ResNet with the same heads. Test predictions on the nine mirror sequences are saved
per window for the fusion evaluation. Writes miti_gru6d/ (state dicts, predictions, results json).

  python train_miti_gru6d.py --models gru6d,ronin_lstm,tlio_resnet --seeds 0,1,2 --epochs 120 --threads 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_inertial_baselines as tib  # noqa: E402
from train_inertial_baselines import EncoderPoseNet, IMUCameraPoseGRU6D  # noqa: E402

OUT = HERE / "miti_gru6d"
D = 6


def build(kind: str):
    if kind == "gru6d":
        return IMUCameraPoseGRU6D(input_dim=D, embed_dim=256, gru_hidden_dim=512, gru_num_layers=2, head_dim=512, head_blocks=2, dropout=0.25)
    return EncoderPoseNet(kind, D=D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gru6d,ronin_lstm,tlio_resnet")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(exist_ok=True)
    tib.build = build
    tib.OUT = OUT
    z = np.load(HERE / "miti_windows.npz", allow_pickle=True)
    X, Y, split, seq = z["X"].astype(np.float32), z["y"].astype(np.float32), z["split"].astype(str), z["test_sequence"].astype(str)
    itr, iva, ite = np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]
    sx = StandardScaler().fit(X[itr].reshape(-1, D))
    sp = StandardScaler().fit(Y[itr])
    N, T, _ = X.shape
    Xs = sx.transform(X.reshape(-1, D)).reshape(N, T, D).astype(np.float32)
    Ys = sp.transform(Y).astype(np.float32)
    tests = [(name, Xs[ite][seq[ite] == name], Y[ite][seq[ite] == name]) for name in sorted(set(seq[ite]))]
    data = (Xs, Ys, itr, iva, tests, sp, None)
    res = {"protocol": __doc__.strip().splitlines()[0], "N_train": int(len(itr)), "N_val": int(len(iva)), "N_test": int(len(ite)),
           "D": D, "epochs": args.epochs, "runs": {}}
    if (OUT / "miti_gru6d_results.json").exists():  # resume: keep finished runs
        res["runs"] = json.loads((OUT / "miti_gru6d_results.json").read_text(encoding="utf-8")).get("runs", {})
    zero_t = {name: float(np.linalg.norm(y[:, 9:12], axis=1).mean()) for name, _, y in tests}
    res["zero_translation_test_mm"] = {"pooled": float(np.linalg.norm(Y[ite][:, 9:12], axis=1).mean()), **zero_t}
    gyro = z["gyro_rot_deg"]
    res["gyro_rotation_test_deg"] = {"pooled": float(np.nanmean(gyro[ite])), **{name: float(np.nanmean(gyro[ite][seq[ite] == name])) for name in zero_t}}
    for kind in args.models.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            if f"{kind}_s{seed}" in res["runs"] and (OUT / f"pred_{kind}_s{seed}.npz").exists():
                print(f"{kind}_s{seed}: already done", flush=True)
                continue
            t0 = time.time()
            rows, n_par, best, secs, val = tib.train_one(kind, seed, data, args.device, args.epochs)
            r_all = np.concatenate([v[0] for v in rows.values()])
            t_all = np.concatenate([v[1] for v in rows.values()])
            # validation-selected shrinkage of the predicted translation (control used in the main paper)
            model = build(kind)
            model.load_state_dict(torch.load(OUT / f"gru6d_s{seed}.pt")) if kind == "gru6d" else None
            key = f"{kind}_s{seed}"
            run = {"params": int(n_par), "best_val_loss": float(best), "train_seconds": float(secs), "validation": val,
                   "test": {"N": int(len(r_all)), "rot_mean": float(r_all.mean()), "rot_p50": float(np.median(r_all)), "rot_p95": float(np.percentile(r_all, 95)),
                            "trans_mean": float(t_all.mean()), "trans_p50": float(np.median(t_all)), "trans_p95": float(np.percentile(t_all, 95))},
                   "per_sequence": {name: {"N": int(len(v[0])), "rot_mean": float(v[0].mean()), "trans_mean": float(v[1].mean())} for name, v in rows.items()}}
            np.savez_compressed(OUT / f"pred_{key}.npz", **{name: v[2] for name, v in rows.items()},
                                **{name + "__rot": v[0] for name, v in rows.items()}, **{name + "__trans": v[1] for name, v in rows.items()})
            res["runs"][key] = run
            print(f"{key}: test rot {run['test']['rot_mean']:.3f} deg, trans {run['test']['trans_mean']:.3f} mm (zero {res['zero_translation_test_mm']['pooled']:.3f}); "
                  f"val rot {val['rot_mean']:.3f} trans {val['trans_mean']:.3f}; {secs:.0f} s", flush=True)
            (OUT / "miti_gru6d_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # validation predictions for the shrinkage factor (GRU6D seed 0)
    if (OUT / "gru6d_s0.pt").exists():
        model = build("gru6d")
        model.load_state_dict(torch.load(OUT / "gru6d_s0.pt", map_location="cpu"))
        pv = sp.inverse_transform(tib.predict(model, Xs[iva], "cpu").astype(np.float64))[:, 9:12]
        yv = Y[iva][:, 9:12]
        grid = np.round(np.arange(0.3, 1.0001, 0.01), 2)
        curve = {float(s): float(np.linalg.norm(s * pv - yv, axis=1).mean()) for s in grid}
        s_val = min(curve, key=curve.get)
        pt = sp.inverse_transform(tib.predict(model, Xs[ite], "cpu").astype(np.float64))[:, 9:12]
        yt = Y[ite][:, 9:12]
        res["shrinkage_gru6d_s0"] = {"s_validation": s_val, "test_trans_s1": float(np.linalg.norm(pt - yt, axis=1).mean()),
                                     "test_trans_shrunk": float(np.linalg.norm(s_val * pt - yt, axis=1).mean()),
                                     "direction_cosine_mean": float(np.mean(np.sum(pt * yt, 1) / (np.linalg.norm(pt, axis=1) * np.linalg.norm(yt, axis=1) + 1e-9))),
                                     "pred_norm_mean": float(np.linalg.norm(pt, axis=1).mean()), "label_norm_mean": float(np.linalg.norm(yt, axis=1).mean())}
        print("shrinkage:", res["shrinkage_gru6d_s0"])
    (OUT / "miti_gru6d_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    import pickle
    pickle.dump({"scaler_imu": sx, "scaler_pose": sp}, open(OUT / "scalers.pkl", "wb"))


if __name__ == "__main__":
    main()
