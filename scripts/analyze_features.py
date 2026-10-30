"""Representativeness report for cached features.

    python scripts/analyze_features.py                       # every cached backbone
    python scripts/analyze_features.py --specs dinov2 srm --skip-umap

Writes outputs/reports/<spec>/{report.html, metrics.json, *.png} and outputs/reports/index.html
(cross-backbone comparison table).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis import viz  # noqa: E402
from ifl.analysis.store import available_specs, load  # noqa: E402

REP = OUTPUTS / "reports"

KEY_COLS = [  # what goes into the cross-backbone table, in order
    ("probe_auc", "IMG probe AUC (real vs fake)"),
    ("probe_auc_label2", "IMG probe AUC (real vs local-edit)"),
    ("lodo_mean_auc", "IMG LODO mean AUC (unseen dataset)"),
    ("format_probe_auc", "IMG predicts JPEG/PNG (confound)"),
    ("probe_auc_within_jpg", "IMG probe AUC within JPEG only"),
    ("dataset_probe_bacc", "IMG predicts dataset (bacc)"),
    ("patch_auc", "PATCH sup. probe AUC"),
    ("img_mean_auc", "PATCH sup. per-image AUC"),
    ("img_mean_f1", "PATCH sup. per-image F1"),
    ("transfer_offdiag_mean", "PATCH transfer AUC (off-diag)"),
    ("det_auc_from_maxpatch", "DET from max patch"),
    ("zs_knn_img_mean_auc", "ZERO-SHOT intra-img kNN AUC"),
    ("zs_proto_img_mean_auc", "ZERO-SHOT intra-img proto AUC"),
    ("oc_img_mean_auc", "ONE-CLASS(real) per-image AUC"),
    ("oc_det_auc_p95", "ONE-CLASS(real) det AUC"),
]


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("-" if v is None else str(v))


def analyze(spec, datasets, skip_umap):
    t0 = time.time()
    out = REP / spec
    out.mkdir(parents=True, exist_ok=True)
    fs = load(spec, datasets)
    print(f"[{spec}] N={len(fs.meta)} D={fs.P.shape[-1]} grid={fs.grid}", flush=True)
    m, figs, tables = {}, [], {}

    # ---- image level
    r = R.probe_image(fs); r.pop("_oof", None); m.update(r)
    lodo = R.lodo_image(fs)
    if len(lodo):
        m["lodo_mean_auc"] = float(lodo.auc.mean()); tables["Leave-one-dataset-out (image level)"] = lodo
    m.update(R.confound_image(fs))
    if not skip_umap:
        viz.umap_global(fs, out / "umap.png"); figs.append(("UMAP of global embeddings", "umap.png"))
    print(f"  image-level done {time.time() - t0:.0f}s", flush=True)

    # ---- patch level
    r, idx, S = R.patch_probe(fs)
    m.update(r)
    maps = {}
    if S is not None:
        lab = R._patch_labels(fs.M)
        maps["sup. linear probe"] = (idx, S[idx])
        viz.score_hist(S[idx], lab[idx], out / "probe_hist.png", f"{spec}: OOF patch-probe scores")
        figs.append(("Supervised patch probe - score distributions", "probe_hist.png"))
        per_ds = {k.split(":")[1]: v for k, v in r.items() if k.endswith(":img_mean_auc")}
        tables["Supervised patch probe per dataset"] = pd.DataFrame(
            {"img_mean_auc": per_ds, "img_mean_f1": {k.split(":")[1]: v for k, v in r.items()
                                                    if k.endswith(":img_mean_f1")}})
    print(f"  patch probe done {time.time() - t0:.0f}s", flush=True)
    mat = R.patch_transfer(fs)
    if len(mat) > 1:
        off = mat.values[~np.eye(len(mat), dtype=bool)]
        m["transfer_offdiag_mean"] = float(np.nanmean(off))
        viz.transfer_heatmap(mat, out / "transfer.png", f"{spec}: per-image patch AUC, train(row) -> test(col)")
        figs.append(("Cross-dataset patch-probe transfer", "transfer.png"))
        tables["Transfer matrix"] = mat
    zs, zmaps = R.intra_image_zero_shot(fs)
    m.update(zs)
    eidx, _ = R._edited(fs)
    maps["zero-shot kNN"] = (eidx, zmaps["knn"])
    oc, Soc = R.one_class_real(fs)
    m.update(oc)
    if Soc is not None:
        maps["one-class (real)"] = (eidx, Soc)
    print(f"  zero-shot / one-class done {time.time() - t0:.0f}s", flush=True)
    if len(eidx):
        viz.patch_gallery(fs, eidx, maps, out / "gallery.png")
        figs.append(("Gallery: image | mask | patch PCA | score maps", "gallery.png"))

    clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
    (out / "metrics.json").write_text(json.dumps(clean, indent=1))
    html = [f"<html><head><meta charset='utf-8'><title>{spec}</title><style>body{{font-family:sans-serif;"
            "margin:24px}} table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid #ccc;"
            "padding:3px 6px}} img{{max-width:100%}}</style></head><body>",
            f"<h1>Feature representativeness - {spec}</h1><p>N={len(fs.meta)}, D={fs.P.shape[-1]}, "
            f"grid={fs.grid}, runtime {time.time() - t0:.0f}s</p><h2>Metrics</h2><table>"]
    html += [f"<tr><td>{k}</td><td>{fmt(v)}</td></tr>" for k, v in clean.items()]
    html.append("</table>")
    for t, df in tables.items():
        html.append(f"<h2>{t}</h2>" + df.to_html(float_format=lambda x: f"{x:.3f}"))
    for t, f in figs:
        html.append(f"<h2>{t}</h2><img src='{f}'>")
    (out / "report.html").write_text("\n".join(html) + "</body></html>", encoding="utf-8")
    print(f"[{spec}] done in {time.time() - t0:.0f}s", flush=True)
    return clean


def summary():
    rows = {}
    for p in sorted(REP.glob("*/metrics.json")):
        rows[p.parent.name] = json.loads(p.read_text())
    if not rows:
        return
    df = pd.DataFrame(rows).T
    cols = [c for c, _ in KEY_COLS if c in df.columns]
    tab = df[cols].rename(columns=dict(KEY_COLS)).astype(float)
    tab.to_csv(REP / "summary.csv")
    html = ["<html><head><meta charset='utf-8'><title>Backbone comparison</title><style>body{font-family:"
            "sans-serif;margin:24px} table{border-collapse:collapse;font-size:12px} td,th{border:1px solid #ccc;"
            "padding:3px 6px;text-align:right}</style></head><body><h1>Latent representativeness - backbone "
            "comparison</h1><p>Rows = frozen feature extractor. 0.5 = chance for every AUC. "
            "<b>pixels</b> is a no-knowledge control: a backbone that does not beat it reads colour/layout, "
            "not forensic traces. Confound rows should be LOW.</p>",
            tab.style.format("{:.3f}", na_rep="-").background_gradient(cmap="RdYlGn", vmin=0.5, vmax=1.0,
                                                                         axis=None).to_html(),
            "<ul>" + "".join(f"<li><a href='{s}/report.html'>{s}</a></li>" for s in tab.index) + "</ul>",
            "</body></html>"]
    (REP / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(tab.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--skip-umap", action="store_true")
    args = ap.parse_args()
    for spec in args.specs or available_specs():
        analyze(spec, args.datasets, args.skip_umap)
    summary()


if __name__ == "__main__":
    main()
