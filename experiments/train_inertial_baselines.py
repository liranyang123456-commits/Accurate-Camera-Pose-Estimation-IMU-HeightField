#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned inertial baselines on the 11-session benchmark under the GRU6D protocol.

Every model sees the same 0.5 s windows (T=60, D=21), the same session-disjoint split
(36 training / 13 validation / 11 test sessions; split_indices.npz of the evaluated GRU6D run),
the same input and pose scalers (taken from the GRU6D checkpoint), the same 6D-rotation and
translation heads, the same loss (8 x geodesic + smooth-L1 translation; TLIO adds a Gaussian
log-likelihood on translation after a 10-epoch warm-up), and the same schedule (AdamW, lr 1e-3,
weight decay 1e-5, batch 64, gradient clipping 1.0, plateau decay x0.5 after 10 epochs, --epochs
(120 in the paper), best-validation checkpoint). Only the temporal encoder differs.

Encoders: GRU6D (re-trained with this loop as a control), RoNIN ResNet-18 1D, RoNIN LSTM (3 x 100),
TLIO ResNet-18 1D with a translation log-variance head. Encoders are adapted to 21-channel 60-sample
windows; results are "re-trained adaptations", not official pretrained scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from paths import IMUCAM_ROOT  # noqa: E402
sys.path.insert(0, str(IMUCAM_ROOT))
from imu_camera_mlp import (  # noqa: E402
    IMUCameraMLPTrainer,
    IMUCameraPoseGRU6D,
    PoseGeodesicLoss,
    _ResidualMLPBlockLN,
    rot6d_to_rotmat_torch,
)
from eval_vio_fixedtest_aggregate import _load_fixed_test_sessions_from_train_config  # noqa: E402

RUN = IMUCAM_ROOT / "out_train_fixed_seq_20260310_031242"
OUT = Path(__file__).resolve().parent / "inertial_baselines"
DS = "fixed_seq_len60_win0.500s_delta_window"


# ----------------------------------------------------------------------------------------- models
class PoseHead(nn.Module):
    def __init__(self, in_dim: int, head_dim: int = 512, blocks: int = 2, dropout: float = 0.25, logvar: bool = False):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, head_dim), nn.GELU(), nn.Dropout(dropout))
        self.rot_blocks = nn.Sequential(*[_ResidualMLPBlockLN(head_dim, expansion=4, dropout=dropout) for _ in range(blocks)])
        self.trans_blocks = nn.Sequential(*[_ResidualMLPBlockLN(head_dim, expansion=4, dropout=dropout) for _ in range(blocks)])
        self.rot_head = nn.Linear(head_dim, 6)
        self.trans_head = nn.Linear(head_dim, 3)
        self.logvar_head = nn.Linear(head_dim, 3) if logvar else None

    def forward(self, feat):
        f = self.shared(feat)
        r6 = self.rot_head(self.rot_blocks(f))
        tf = self.trans_blocks(f)
        pose12 = torch.cat([rot6d_to_rotmat_torch(r6).reshape(-1, 9), self.trans_head(tf)], dim=1)
        if self.logvar_head is not None:
            return pose12, torch.clamp(self.logvar_head(tf), -8.0, 4.0)
        return pose12


class BasicBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm1d(cout))
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        return F.relu(self.net(x) + self.skip(x))


class ResNet18_1D(nn.Module):
    """RoNIN / TLIO ResNet-18 1D backbone (stem 7/2 + max-pool, stages 64-128-256-512, 2 blocks each)."""

    def __init__(self, cin: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool1d(3, 2, 1))
        layers, c = [], 64
        for stage, cout in enumerate((64, 128, 256, 512)):
            for b in range(2):
                layers.append(BasicBlock1D(c, cout, 2 if (stage > 0 and b == 0) else 1))
                c = cout
        self.body = nn.Sequential(*layers)

    def forward(self, x):  # x: (B,T,D)
        return F.adaptive_avg_pool1d(self.body(self.stem(x.transpose(1, 2))), 1).squeeze(-1)


class EncoderPoseNet(nn.Module):
    def __init__(self, kind: str, D: int = 21):
        super().__init__()
        self.kind = kind
        if kind in ("ronin_resnet", "tlio_resnet"):
            self.enc, fdim = ResNet18_1D(D), 512
        elif kind == "ronin_lstm":
            self.enc, fdim = nn.LSTM(D, 100, num_layers=3, batch_first=True, dropout=0.2), 100
        else:
            raise ValueError(kind)
        self.head = PoseHead(fdim, logvar=(kind == "tlio_resnet"))

    def forward(self, x):
        if self.kind == "ronin_lstm":
            _, (h, _) = self.enc(x)
            feat = h[-1]
        else:
            feat = self.enc(x)
        return self.head(feat)


def build(kind: str):
    if kind == "gru6d":
        return IMUCameraPoseGRU6D(input_dim=21, embed_dim=256, gru_hidden_dim=512, gru_num_layers=2,
                                  head_dim=512, head_blocks=2, dropout=0.25)
    return EncoderPoseNet(kind)


# ------------------------------------------------------------------------------------------- data
def load_data():
    cfg = json.loads((RUN / "train_config.json").read_text(encoding="utf-8"))
    z = np.load(RUN / "split_indices.npz")
    X = np.concatenate([np.load(Path(d["dataset_dir"]) / "X.npy") for d in cfg["datasets"]]).astype(np.float32)
    Y = np.concatenate([np.load(Path(d["dataset_dir"]) / "y.npy") for d in cfg["datasets"]]).astype(np.float32)[:, :12]
    assert len(X) == cfg["merged_shape"]["N"]
    ref = IMUCameraMLPTrainer.from_checkpoint(str(RUN / "best_model.pth"), device="cpu")
    sx, sp = ref.scaler_imu, ref.scaler_pose
    N, T, D = X.shape
    Xs = sx.transform(X.reshape(-1, D)).reshape(N, T, D).astype(np.float32)
    Ys = sp.transform(Y).astype(np.float32)
    test_sessions = _load_fixed_test_sessions_from_train_config(str(RUN / "train_config.json"))
    tests = []
    for sd in test_sessions:
        x = np.load(Path(sd) / DS / "X.npy").astype(np.float32)
        y = np.load(Path(sd) / DS / "y.npy").astype(np.float32)[:, :12]
        tests.append((Path(sd).name, sx.transform(x.reshape(-1, D)).reshape(x.shape).astype(np.float32), y))
    return Xs, Ys, z["idx_train"], z["idx_val"], tests, sp, ref


def project_so3(R):
    U, _, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(U @ Vt))
    U[:, :, -1] *= d[:, None]
    return U @ Vt


def errors(pred_s, y, sp):
    p = sp.inverse_transform(pred_s.astype(np.float64))
    Rp = project_so3(p[:, :9].reshape(-1, 3, 3))
    Rt = project_so3(y[:, :9].astype(np.float64).reshape(-1, 3, 3))
    c = (np.einsum("nij,nij->n", Rp, Rt) - 1.0) * 0.5
    rot = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    trans = np.linalg.norm(p[:, 9:12] - y[:, 9:12], axis=1)
    return rot, trans


def predict(model, X, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(torch.from_numpy(X[i:i + bs]).to(device))
            out.append((o[0] if isinstance(o, (tuple, list)) else o).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------------------- training
def train_one(kind, seed, data, device, epochs):
    Xs, Ys, itr, iva, tests, sp, _ = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(kind).to(device)
    crit = PoseGeodesicLoss(rot_weight=8.0, trans_weight=1.0, trans_loss="smoothl1").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5, min_lr=1e-6)
    Xtr, Ytr = torch.from_numpy(Xs[itr]).to(device), torch.from_numpy(Ys[itr]).to(device)
    Xva, Yva = torch.from_numpy(Xs[iva]).to(device), torch.from_numpy(Ys[iva]).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, t0 = float("inf"), None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64].to(device)
            out = model(Xtr[b])
            if isinstance(out, (tuple, list)):
                pose, logvar = out
                if ep < 10:
                    loss = crit(pose, Ytr[b])
                else:
                    Rp, Rt = pose[:, :9].reshape(-1, 3, 3), Ytr[b][:, :9].reshape(-1, 3, 3)
                    c = torch.clamp(((Rp.transpose(1, 2) @ Rt).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2, -1 + 1e-6, 1 - 1e-6)
                    e2 = (pose[:, 9:12] - Ytr[b][:, 9:12]) ** 2
                    loss = 8.0 * torch.acos(c).mean() + (0.5 * (e2 * torch.exp(-logvar) + logvar)).mean()
            else:
                loss = crit(out, Ytr[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(Xva), 64):
                o = model(Xva[i:i + 64])
                vl.append(float(crit(o[0] if isinstance(o, (tuple, list)) else o, Yva[i:i + 64])))
            v = float(np.mean(vl))
        sch.step(v)
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"  [{kind} s{seed}] ep {ep:4d} val {v:.4f} best {best:.4f} lr {opt.param_groups[0]['lr']:.1e} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    rows = {}
    for name, x, y in tests:
        r, t = errors(predict(model, x, device), y, sp)
        rows[name] = (r, t)
    n_par = sum(p.numel() for p in model.parameters())
    return rows, n_par, best, time.time() - t0


def summarize(rows):
    r = np.concatenate([v[0] for v in rows.values()])
    t = np.concatenate([v[1] for v in rows.values()])
    return {"N": int(len(r)), "rot_mean": float(r.mean()), "rot_p50": float(np.median(r)), "rot_p95": float(np.percentile(r, 95)),
            "trans_mean": float(t.mean()), "trans_p50": float(np.median(t)), "trans_p95": float(np.percentile(t, 95)),
            "hit_rot5": float(np.mean(r <= 5.0)), "hit_trans20": float(np.mean(t <= 20.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ronin_resnet,ronin_lstm,tlio_resnet,gru6d")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = True
    res_path = OUT / f"inertial_baselines_results_s{args.seeds.replace(',', '')}.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    _, _, _, _, tests, sp, ref = data
    ref.model.to(device)
    ref_rows = {name: errors(predict(ref.model, x, device), y, sp) for name, x, y in tests}
    res = {"protocol": __doc__.strip().splitlines()[0], "reference_gru6d_checkpoint": summarize(ref_rows), "runs": {}}
    np.savez(OUT / "pred_errors_reference_gru6d.npz", **{f"{k}__rot": v[0] for k, v in ref_rows.items()},
             **{f"{k}__trans": v[1] for k, v in ref_rows.items()})
    print("reference GRU6D checkpoint:", res["reference_gru6d_checkpoint"], flush=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for kind in args.models.split(","):
            rows, n_par, best, secs = train_one(kind, seed, data, device, args.epochs)
            s = summarize(rows)
            ses = [k for k in ref_rows]
            d_rot = np.array([rows[k][0].mean() - ref_rows[k][0].mean() for k in ses])
            d_tr = np.array([rows[k][1].mean() - ref_rows[k][1].mean() for k in ses])
            s.update({"params": int(n_par), "best_val_loss": best, "train_seconds": round(secs, 1),
                      "sessions_worse_than_gru6d_rot": int((d_rot > 0).sum()),
                      "session_wilcoxon_p_rot_vs_gru6d": float(stats.wilcoxon(d_rot).pvalue),
                      "sessions_worse_than_gru6d_trans": int((d_tr > 0).sum()),
                      "session_wilcoxon_p_trans_vs_gru6d": float(stats.wilcoxon(d_tr).pvalue)})
            res["runs"][f"{kind}_s{seed}"] = s
            np.savez(OUT / f"pred_errors_{kind}_s{seed}.npz", **{f"{k}__rot": v[0] for k, v in rows.items()},
                     **{f"{k}__trans": v[1] for k, v in rows.items()})
            res_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"== {kind} seed {seed}: rot {s['rot_mean']:.3f} deg, trans {s['trans_mean']:.3f} mm, "
                  f"params {n_par/1e6:.2f}M, {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
