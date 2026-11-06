"""Tables from the decoder-trace harness rows (outputs/comparator/rows_dtrace<suffix>.csv, written by eval_dtrace.py):
mean per-image cell AUC per scorer and dataset, matched null, AUC - null, edit families.

    python scripts/summarize_dtrace.py [__jpeg75]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402

FAMILY = {
    "regenerated / LLM edit (published)": ["gpt_image_2", "nanobanana2", "tgif2_sd2_fr", "tgif2_sdxl_fr",
                                           "tgif2_flux1dev_fr"],
    "diffusion splice / inpaint (published)": ["tgif2_sd2_sp", "tgif2_flux1dev_sp", "cocoglide", "diverse_inpaint",
                                               "defacto_inpaint", "imd2020_inpaint", "opensdi", "sid_set",
                                               "diffseg30k", "autosplice"],
    "manual / classical": ["nc2016", "korus", "imd2020_real", "cmfd_velo"],
    "benchmark regenerated (fr)": ["bench_sdxlinp_fr", "bench_fluxfill_fr", "bench_kontext_fr"],
    "benchmark spliced (sp)": ["bench_sdxlinp_sp", "bench_fluxfill_sp", "bench_kontext_sp"],
    "extension regenerated (fr)": ["benchx_sdxlinp_fr", "benchx_fluxfill_fr", "benchx_kontext_fr"],
    "extension spliced (sp)": ["benchx_sdxlinp_sp", "benchx_fluxfill_sp", "benchx_kontext_sp"],
    "GPT-Image-2": ["gpt_image_2"],
    "Nano-Banana-2": ["nanobanana2"],
    "TGIF2 regenerated (fr)": ["tgif2_sd2_fr", "tgif2_sdxl_fr", "tgif2_flux1dev_fr"],
}
WITH_NULL = ("benchmark", "extension", "GPT", "Nano", "TGIF2")     # families whose images have matched originals
MAIN = ["x_trufor", "x_mesorch", "cmp_z2", "dtA", "dtB", "dtB_max", "sota+cmp", "dtB+cmp", "sota+cmp+dtB",
        "PIX_loc_cf"]
FLAME = ["x_flame", "x_flame_max", "dtB+flame", "sota+cmp+flame", "sota+cmp+dtB+flame"]
BFREE = ["x_bfree", "x_bfree_max", "dtB+bfree", "sota+cmp+dtB+bfree", "sota+cmp+dtB+flame+bfree"]


def main(suffix="", tag="dtrace"):
    d = pd.read_csv(OUTPUTS / "comparator" / f"rows_{tag}{suffix}.csv")
    main_ = MAIN + [s for s in FLAME + BFREE if s in set(d.scorer)]
    d = d[d.scorer.isin(main_)]
    e, a = d[d.kind == "edited"], d[d.kind == "authentic"]
    null_dom = a.groupby(["scorer", "domain"]).auc.mean()
    rows = []
    for sc in main_:
        es = e[e.scorer == sc]
        r = {"scorer": sc, "ALL edited": es.auc.mean(), "interior": es.auc_int.mean(),
             "null (all matched)": a[a.scorer == sc].auc.mean()}
        r["ALL - null"] = r["ALL edited"] - r["null (all matched)"]
        for f, ds in FAMILY.items():
            g = es[es.dataset.isin(ds)]
            r[f] = g.auc.mean()
            nulls = [null_dom.get((sc, dm)) for dm in set(g.domain) if (sc, dm) in null_dom.index]
            if nulls and f.startswith(WITH_NULL):
                r[f + " - null"] = r[f] - sum(nulls) / len(nulls)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("scorer").T.round(3)
    per = e.pivot_table(index="dataset", columns="scorer", values="auc")[main_].round(3)
    nul = a.pivot_table(index="domain", columns="scorer", values="auc")[main_].round(3)
    md = (f"# Decoder-trace detector vs published localizers - localization, condition '{suffix or 'clean'}' "
          "(mean per-image cell AUC)\n\n"
          "dtA: trained on 5 autoencoders (SD1.5, SDXL, FLUX, MoVQ, Qwen-Image), dtB: on all 10. Matched null = the "
          "same map on the edit's own original with the edit's mask (TGIF2, benchmark, extension, GPT-Image-2 and "
          "Nano-Banana-2 with their COCO originals, Korus, NC2016, IMD2020, CMFD); OpenSDI / SID-Set: random masks "
          "of the same source.\n\n## Summary\n\n" + t.to_markdown()
          + "\n\n## Per dataset (edited images)\n\n" + per.to_markdown()
          + "\n\n## Nulls per source (authentic images, transplanted masks)\n\n" + nul.to_markdown() + "\n")
    out = OUTPUTS / "dtrace"
    out.mkdir(parents=True, exist_ok=True)
    name = {"dtrace": "localization", "dtflame": "localization_flame"}.get(tag, f"localization_{tag}")
    (out / f"{name}{suffix}.md").write_text(md)
    t.to_csv(out / f"{name}_summary{suffix}.csv")
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "", sys.argv[2] if len(sys.argv) > 2 else "dtrace")
