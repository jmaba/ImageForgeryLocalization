"""Visual tools for inspecting latent representativeness."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ifl import DATA, LABEL_NAMES  # noqa: E402


def _cat_colors(values):
    cats = sorted(set(values), key=str)
    cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    return {c: cmap(i % cmap.N) for i, c in enumerate(cats)}


def umap_global(fs, path, max_n=4000):
    meta = fs.meta
    sel = np.random.default_rng(0).permutation(len(meta))[:max_n]
    X = StandardScaler().fit_transform(fs.G[sel])
    if X.shape[1] > 50:
        X = PCA(50, random_state=0).fit_transform(X)
    try:
        import umap
        E = umap.UMAP(n_neighbors=20, min_dist=0.2, metric="cosine", random_state=0).fit_transform(X)
        proj = "UMAP"
    except ImportError:  # numba lags behind numpy releases; t-SNE needs nothing extra
        from sklearn.manifold import TSNE
        E = TSNE(perplexity=30, metric="cosine", init="pca", random_state=0).fit_transform(X)
        proj = "t-SNE"
    cols = [("label", meta.label.map(LABEL_NAMES).values[sel]), ("dataset", meta.dataset.values[sel]),
            ("format", meta.format.values[sel])]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (title, v) in zip(axs, cols):
        cmap = _cat_colors(v)
        for c, col in cmap.items():
            m = v == c
            ax.scatter(E[m, 0], E[m, 1], s=5, color=col, label=f"{c} ({m.sum()})", alpha=0.7)
        ax.set_title(f"{proj} of global features - colour = {title}")
        ax.legend(markerscale=3, fontsize=7, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(fs.spec)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _pca_rgb(P):
    g, _, D = P.shape
    Z = P.reshape(-1, D).astype(np.float32)
    if D >= 3:
        Z = PCA(3, random_state=0).fit_transform(StandardScaler().fit_transform(Z))
    lo, hi = np.percentile(Z, 2, 0), np.percentile(Z, 98, 0)
    return np.clip((Z - lo) / (hi - lo + 1e-8), 0, 1).reshape(g, g, -1)[..., :3]


def patch_gallery(fs, idx_rows, maps: dict, path, n_per_ds=3, seed=0):
    """Rows: image | GT mask | patch-PCA->RGB | one panel per score map in `maps`.
    maps: name -> (indices_into_meta, array [n,g,g])"""
    rng = np.random.default_rng(seed)
    pick = []
    for ds in sorted(fs.meta.dataset.iloc[idx_rows].unique()):
        cand = idx_rows[(fs.meta.dataset.iloc[idx_rows] == ds).values]
        pick += list(rng.choice(cand, min(n_per_ds, len(cand)), replace=False))
    ncol = 3 + len(maps)
    fig, axs = plt.subplots(len(pick), ncol, figsize=(2.3 * ncol, 2.3 * len(pick)), squeeze=False)
    lookup = {k: {int(j): n for n, j in enumerate(ii)} for k, (ii, _) in maps.items()}
    for r, i in enumerate(pick):
        row = fs.meta.iloc[i]
        im = Image.open(DATA / row.img_path).convert("RGB").resize((256, 256))
        axs[r, 0].imshow(im); axs[r, 0].set_ylabel(f"{row.dataset}\n{str(row.sample_id)[:18]}", fontsize=7)
        axs[r, 1].imshow(fs.M[i], vmin=0, vmax=1, cmap="gray")
        axs[r, 2].imshow(_pca_rgb(fs.P[i]))
        for c, (name, (_, arr)) in enumerate(maps.items()):
            ax = axs[r, 3 + c]
            if i in lookup[name]:
                ax.imshow(arr[lookup[name][i]], cmap="inferno")
            if r == 0:
                ax.set_title(name, fontsize=8)
        if r == 0:
            for c, t in enumerate(["image", "GT mask (grid)", "patch PCA->RGB"]):
                axs[r, c].set_title(t, fontsize=8)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(fs.spec, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def transfer_heatmap(mat, path, title):
    fig, ax = plt.subplots(figsize=(1.1 * len(mat) + 2, 0.9 * len(mat) + 1.5))
    v = mat.values.astype(float)
    im = ax.imshow(v, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
    for a in range(v.shape[0]):
        for b in range(v.shape[1]):
            ax.text(b, a, f"{v[a, b]:.2f}", ha="center", va="center", color="w" if v[a, b] < 0.8 else "k",
                    fontsize=8)
    ax.set_xlabel("test dataset"); ax.set_ylabel("train dataset")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def score_hist(S, lab, path, title):
    v = lab >= 0
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for c, name in [(0, "authentic"), (1, "manipulated")]:
        ax.hist(S[v & (lab == c)], bins=80, alpha=0.55, density=True, label=name)
    ax.legend(); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def per_dataset_bar(df, path, title):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df.columns) + 3), 4))
    df.T.plot.bar(ax=ax, width=0.85)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_ylim(0.3, 1.0); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
