"""Fig. 3: GRU6D as evaluated (layer sizes of the configuration in train_inertial_baselines.build / the checkpoint config).
Output: experiments/figures/Fig3_gru6d.pdf and .png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
FILL = {"in": "#e8eef7", "enc": "#d6e6f2", "shared": "#f3e3c7", "rot": "#e7dff2", "trans": "#d9efe2", "out": "#f2f2f2"}
W, H = 1.75, 1.05


def box(ax, x, y, title, sub, kind, w=W, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.015,rounding_size=0.05", fc=FILL[kind], ec="#2e6b47" if bold else "#4a4a4a", lw=1.7 if bold else 0.9))
    ax.text(x + w / 2, y + H * 0.70, title, ha="center", va="center", fontsize=7.4,
            linespacing=1.05)
    ax.text(x + w / 2, y + H * 0.27, sub, ha="center", va="center", fontsize=6.0, color="#333333", linespacing=1.1)


def arrow(ax, a, b, color="#3d6e7a"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(-0.35, 4.55)
    ax.axis("off")
    ym, yr, yt = 1.70, 3.15, 0.25
    xs = [0.05, 2.10, 4.15, 6.20]
    box(ax, xs[0], ym, "IMU window", "60 x 21,\nshifted by $\\delta$", "in")
    box(ax, xs[1], ym, "Linear +\nLayerNorm", "60 x 256", "enc")
    box(ax, xs[2], ym, "GRU,\n2 layers", "512 hidden units", "enc")
    box(ax, xs[3], ym, "Shared layer", "LayerNorm, Linear,\nGELU: 512", "shared")
    for k in range(3):
        arrow(ax, (xs[k] + W, ym + H / 2), (xs[k + 1], ym + H / 2))
    ax.text((xs[2] + W + xs[3]) / 2, ym + H / 2 + 0.12, "$\\mathbf{h}_T$", ha="center", va="bottom", fontsize=7.5)
    xh, xo, xp = 8.45, 10.75, 12.80
    box(ax, xh, yr, "Rotation head", "2 residual MLP blocks\n(512), Linear: 6", "rot", w=2.0)
    box(ax, xo, yr, "Gram-Schmidt", "6D to SO(3):\n$\\hat{\\mathbf{R}}$ (3 x 3)", "rot")
    box(ax, xh, yt, "Translation head", "2 residual MLP blocks\n(512), Linear: 3", "trans", w=2.0, bold=True)
    box(ax, xo, yt, "Translation", "$\\hat{\\mathbf{t}}$ (3, mm)", "trans")
    box(ax, xp, ym, "Pose", "[$\\hat{\\mathbf{R}}$ | $\\hat{\\mathbf{t}}$]: 12", "out")
    arrow(ax, (xs[3] + W, ym + H / 2), (xh, yr + H / 2))
    arrow(ax, (xs[3] + W, ym + H / 2), (xh, yt + H / 2))
    arrow(ax, (xh + 2.0, yr + H / 2), (xo, yr + H / 2))
    arrow(ax, (xh + 2.0, yt + H / 2), (xo, yt + H / 2))
    arrow(ax, (xo + W, yr + H / 2), (xp + W / 2, ym + H))
    arrow(ax, (xo + W, yt + H / 2), (xp + W / 2, ym))
    ax.text(xh + 1.0, yt - 0.08, "used as the translation of the inertial increment", ha="center", va="top", fontsize=6.4, color="#2e6b47")
    ax.text(xh + 2.0, yr + H + 0.08, "loss: $8\\,\\mathcal{L}_R$ (geodesic) $+\\ \\mathcal{L}_t$ (Huber); 11.44 M parameters", ha="center",
            va="bottom", fontsize=6.4, color="#555555")
    fig.savefig(OUT / "Fig3_gru6d.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig3_gru6d.png", bbox_inches="tight", dpi=220)
    print("wrote", OUT / "Fig3_gru6d.pdf")


if __name__ == "__main__":
    main()
