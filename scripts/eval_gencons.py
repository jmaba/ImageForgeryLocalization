"""Pilot evaluation of the generator-consistency maps (scripts/remote/gen_consistency.py), docs/07.

Joins the maps with the clean expert dump (outputs/ensemble/dump_phys.pkl) for the pilot domains (TGIF2, GPT-Image-2,
Nano-Banana-2, CocoGlide) and reports
  1. single maps: mean per-image cell AUC per dataset (score = minus the error: generated content is predicted to
     reconstruct / denoise better), with three nulls on the same transplanted masks:
       matched   TGIF2 originals (the edit's own source image)
       coco      the COCO original of each GPT-Image-2 / Nano-Banana-2 edit
       vae_sd2 / vae_flux   TGIF2 originals re-decoded by that autoencoder (benign processing control)
  2. image level (the "original / fully generated / locally edited" triage): AUROC of the image-mean error between
     authentic images and each edited dataset, and against the re-decoded controls
  3. stacked ensemble (scripts/ensemble.py, leave-one-domain-out over the pilot domains) with and without the maps
Maps entering 1 and 3 are fixed in advance: per-model maps, plus the minimum over models (AEROBLADE takes the
minimum LPIPS distance over autoencoders).

    python scripts/eval_gencons.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble as E  # noqa: E402
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402

GC = DATA / "features" / "gencons"
DOMAINS = {"tgif2", "gpt_image_2", "nanobanana2", "cocoglide"}


def load_gc(name):
    z = np.load(GC / f"{name}.npz", allow_pickle=True)   # our own files (ids may be stored as objects)
    names = list(z["names"])
    return {k: p.astype(np.float32).reshape(-1, len(names)) for k, p in zip(z["ids"], z["patch"])}, names


def derived(P, names):
    """[576, F] raw maps -> dict of scores (higher = more likely generated here)"""
    col = {n: P[:, i] for i, n in enumerate(names)}
    out = {f"gc_{n}": -col[n] for n in names}
    for kind in ("ae_l2", "ae_lp"):
        out[f"gc_{kind}_min"] = -np.min([col[n] for n in names if n.startswith(kind + "_")], 0)
    for t in (100, 300, 500):
        out[f"gc_den_t{t}_min"] = -np.min([col[f"den_{m}_t{t}"] for m in ("sd2", "sdxl")], 0)
    return out


def auc_job(j, s):
    l = j["l"].astype(int)
    v = l >= 0
    return R._auc(l[v], s[v]) if (l[v] == 1).any() and (l[v] == 0).any() else np.nan


def main():
    global GC
    if len(sys.argv) > 1:                                    # optional: another folder of maps (e.g. a smoke run)
        GC = Path(sys.argv[1])
    dump =[j for j in pickle.load(open(OUTPUTS / "ensemble" / "dump_phys.pkl", "rb")) if j["domain"] in DOMAINS]
    feats, names = {}, None
    for f in GC.glob("*.npz"):
        feats[f.stem], names = load_gc(f.stem)
    jobs, extra = [], []
    for j in dump:
        ds, key = j["dataset"], j["sample"].split("/", 1)[1]
        if key not in feats.get(ds, {}):
            continue
        j = dict(j, scores={**j["scores"], **derived(feats[ds][key], names)})
        jobs.append(j)
        if ds == "tgif2_orig":                               # same transplanted mask, re-decoded image
            for c in ("vae_sd2", "vae_flux"):
                if key in feats.get(f"tgif2_orig__{c}", {}):
                    extra.append(dict(j, dataset=f"ctrl_{c}", kind="control",
                                      scores=derived(feats[f"tgif2_orig__{c}"][key], names)))
        if ds in ("gpt_image_2", "nanobanana2") and key in feats.get(f"coco_orig_{ds}", {}):
            extra.append(dict(j, dataset=f"coco_orig_{ds}", kind="coco_null",
                              scores=derived(feats[f"coco_orig_{ds}"][key], names)))
    gcs = [k for k in jobs[0]["scores"] if k.startswith("gc_")]
    print(f"{len(jobs)} jobs with maps, {len(extra)} control / coco-null jobs, {len(gcs)} maps")

    # 1. single maps
    rows = []
    for j in jobs + extra:
        for k in gcs + ["x_mesorch", "x_trufor", "cmp_z2"]:
            if k in j["scores"]:
                rows.append(dict(dataset=j["dataset"], kind=j["kind"], scorer=k, auc=auc_job(j, j["scores"][k])))
    r = pd.DataFrame(rows)
    t = r.pivot_table(index="scorer", columns="dataset", values="auc").round(3)
    fr = ["tgif2_sd2_fr", "tgif2_sdxl_fr", "tgif2_flux1dev_fr", "gpt_image_2", "nanobanana2"]
    sp = ["tgif2_sd2_sp", "tgif2_flux1dev_sp", "cocoglide"]
    t["REGEN mean"] = t[[c for c in fr if c in t]].mean(1).round(3)
    t["SPLICE mean"] = t[[c for c in sp if c in t]].mean(1).round(3)
    order = ["REGEN mean", "SPLICE mean"] + fr + sp + ["tgif2_orig", "ctrl_vae_sd2", "ctrl_vae_flux",
                                                       "coco_orig_gpt_image_2", "coco_orig_nanobanana2"]
    t = t[[c for c in order if c in t]].sort_values("REGEN mean", ascending=False)

    # 2. image level: authentic vs each set, from the image-mean raw error (lower = generated)
    img = []
    for ds, fd in feats.items():
        for key, P in fd.items():
            img.append(dict(dataset=ds, **{n: P[:, i].mean() for i, n in enumerate(names)}))
    img = pd.DataFrame(img)
    auth = img[img.dataset.isin(["tgif2_orig", "coco_orig_gpt_image_2", "coco_orig_nanobanana2"])]
    lv = {}
    for ds in sorted(set(img.dataset) - set(auth.dataset)):
        e = img[img.dataset == ds]
        lv[ds] = {n: R._auc(np.r_[np.zeros(len(auth)), np.ones(len(e))], -np.r_[auth[n], e[n]]) for n in names}
    lv = pd.DataFrame(lv).round(3)

    # 3. stacked ensemble, leave-one-domain-out over the pilot domains
    base = {"mesorch": "x_mesorch", "trufor": "x_trufor", "comparator": "cmp_z2"}
    gsel = {"gc_ae_lp": "gc_ae_lp_min", "gc_ae_l2": "gc_ae_l2_min", "gc_den100": "gc_den_t100_min",
            "gc_den300": "gc_den_t300_min", "gc_den500": "gc_den_t500_min"}
    E.EXPERTS.update(base | gsel)
    ens = {}
    for tag, nm in (("stack: mesorch+trufor+comparator", list(base)),
                    ("stack: + generator consistency", list(base) + list(gsel)),
                    ("stack: generator consistency only", list(gsel)),
                    ("zsum: mesorch+trufor+comparator", None)):
        maps = [E.expert_z(j, list(base)).sum(1) for j in jobs] if nm is None else E.stack_maps(jobs, jobs, nm)
        m = E.metrics(jobs, maps)
        e, a = m[m.kind == "edited"], m[m.kind == "authentic"]
        ens[tag] = {"ALL": e.auc.mean(), "null (TGIF2 orig)": a.auc.mean(),
                    "REGEN": e[e.dataset.isin(fr)].auc.mean(), "SPLICE": e[e.dataset.isin(sp)].auc.mean(),
                    **{ds: e[e.dataset == ds].auc.mean() for ds in fr + sp}}
    ens = pd.DataFrame(ens).round(3)

    out = OUTPUTS / "gencons"
    out.mkdir(parents=True, exist_ok=True)
    md = (f"# Generator-consistency pilot (clean)\n\n## 1. Single maps: mean per-image cell AUC\n\n{t.to_markdown()}\n\n"
          f"## 2. Image level: AUROC authentic vs set (image-mean error, lower = generated)\n\n{lv.to_markdown()}\n\n"
          f"## 3. Stacked ensemble, leave-one-domain-out over the pilot domains\n\n{ens.to_markdown()}\n")
    (out / "results_pilot.md").write_text(md)
    r.to_csv(out / "rows_pilot.csv", index=False)
    print(md)


if __name__ == "__main__":
    main()
