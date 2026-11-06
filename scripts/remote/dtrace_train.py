"""Train the decoder-trace detector (ifl/features/dtrace.py, docs/09) on content-matched splices built on the fly:
a real photo crop, its round trip through one autoencoder of the training split, pasted back inside a random blob
(or everywhere, or nowhere), both stored as 8-bit, sometimes re-encoded as JPEG or resized. Label = the blob.

Photos: COCO train2017 (seeded sample, short side >= 400), excluding every COCO id used by TGIF2, GPT-Image-2 and
Nano-Banana-2; the first 300 are validation photos. Validation every 500 steps on all ten autoencoders, so the
held-out decoders of split A show how far the traces generalise.

    python scripts/remote/dtrace_train.py prepare
    python scripts/remote/dtrace_train.py train --splits A B
"""
import argparse
import io
import json
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.features.dtrace import BANK, SPLITS, TraceNet, load_vae, roundtrip  # noqa: E402

SEED, CROP, N_VAL = 2026, 384, 300
CACHE = DATA / "_cache" / "coco_train"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def prepare(n=4300):
    CACHE.mkdir(parents=True, exist_ok=True)
    cap = CACHE / "captions_train2017.json"
    if not cap.exists():
        with urllib.request.urlopen(ANN_URL, timeout=900) as r:
            cap.write_bytes(zipfile.ZipFile(io.BytesIO(r.read())).read("annotations/captions_train2017.json"))
    imgs = json.loads(cap.read_text())["images"]
    m = pd.read_csv(DATA / "samples" / "manifest.csv", dtype=str)
    used = {int(s.split("_")[0]) for s in m[m.dataset.str.startswith("tgif2")].sample_id if s.split("_")[0].isdigit()}
    used |= {int(p.stem) for p in (DATA / "pilot_extra" / "coco_orig").glob("*.jpg")}
    ok = sorted(i["id"] for i in imgs if min(i["width"], i["height"]) >= 400 and i["id"] not in used)
    random.Random(SEED).shuffle(ok)
    ids = ok[:n]

    def get(cid):
        p = CACHE / f"{cid:012d}.jpg"
        if not p.exists():
            for _ in range(3):
                try:
                    with urllib.request.urlopen(f"http://images.cocodataset.org/train2017/{cid:012d}.jpg",
                                                timeout=60) as r:
                        p.write_bytes(r.read())
                    break
                except Exception:
                    time.sleep(2)
        return p.exists()
    with ThreadPoolExecutor(16) as ex:
        got = list(ex.map(get, ids))
    ids = [c for c, g in zip(ids, got) if g]
    (CACHE / "split.json").write_text(json.dumps({"val": ids[:N_VAL], "train": ids[N_VAL:]}))
    print(f"{len(ids)} photos ({N_VAL} validation), {len(used)} COCO ids excluded", flush=True)


class Crops(torch.utils.data.Dataset):
    def __init__(self, ids, seed=None):
        self.ids, self.seed = ids, seed

    def __len__(self):
        return len(self.ids) if self.seed is not None else 10 ** 7

    def __getitem__(self, i):
        rng = random.Random(self.seed + i) if self.seed is not None else random
        cid = self.ids[i % len(self.ids)] if self.seed is not None else rng.choice(self.ids)
        im = Image.open(CACHE / f"{cid:012d}.jpg").convert("RGB")
        if min(im.size) < CROP:
            s = CROP / min(im.size)
            im = im.resize((round(im.width * s) + 1, round(im.height * s) + 1), Image.BICUBIC)
        x0, y0 = rng.randrange(im.width - CROP + 1), rng.randrange(im.height - CROP + 1)
        a = np.asarray(im.crop((x0, y0, x0 + CROP, y0 + CROP))).copy()
        if rng.random() < 0.5:
            a = a[:, ::-1].copy()
        return torch.from_numpy(a).permute(2, 0, 1)


def blobs(B, g, p_none=0.15, p_full=0.15):
    """soft masks [B,1,CROP,CROP]: random unions of distorted ellipses, some full, some empty, some feathered"""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, CROP, device="cuda"), torch.linspace(-1, 1, CROP, device="cuda"),
                            indexing="ij")
    out = torch.zeros(B, 1, CROP, CROP, device="cuda")
    for b in range(B):
        u = torch.rand(1, generator=g, device="cuda").item()
        if u < p_none:
            continue
        if u < p_none + p_full:
            out[b] = 1
            continue
        field = torch.zeros(CROP, CROP, device="cuda")
        for _ in range(int(torch.randint(1, 4, (1,), generator=g, device="cuda"))):
            cx, cy, th = (torch.rand(3, generator=g, device="cuda") * torch.tensor([1.6, 1.6, 3.14], device="cuda")
                          - torch.tensor([0.8, 0.8, 0], device="cuda"))
            ax, ay = 0.12 + 0.5 * torch.rand(2, generator=g, device="cuda")
            xr = (xx - cx) * torch.cos(th) + (yy - cy) * torch.sin(th)
            yr = -(xx - cx) * torch.sin(th) + (yy - cy) * torch.cos(th)
            field = torch.maximum(field, 1 - (xr / ax) ** 2 - (yr / ay) ** 2)
        noise = F.interpolate(torch.randn(1, 1, 8, 8, generator=g, device="cuda"), size=(CROP, CROP),
                              mode="bicubic", align_corners=False)[0, 0]
        m = (field + 0.25 * noise > 0).float()
        if torch.rand(1, generator=g, device="cuda").item() < 0.5:          # feathered paste
            s = 0.5 + 2.0 * torch.rand(1, generator=g, device="cuda").item()
            k = torch.arange(-6, 7, device="cuda", dtype=torch.float32)
            k = torch.exp(-k ** 2 / (2 * s * s))
            k = k / k.sum()
            m = F.conv2d(F.conv2d(m[None, None], k.view(1, 1, 1, -1), padding=(0, 6)), k.view(1, 1, -1, 1),
                         padding=(6, 0))[0, 0]
        out[b, 0] = m
    return out


def q8(x):
    return torch.round((x + 1) * 127.5).clamp(0, 255) / 127.5 - 1


def augment(x, g):
    """x [B,3,H,W] in [-1,1] on GPU -> JPEG (p 0.3, quality 75-100) and/or down-up resize (p 0.15)"""
    from torchvision.io import decode_jpeg, encode_jpeg
    out = []
    for xi in x:
        if torch.rand(1, generator=g, device="cuda").item() < 0.15:
            s = 0.6 + 0.35 * torch.rand(1, generator=g, device="cuda").item()
            h = max(32, int(CROP * s))
            xi = F.interpolate(F.interpolate(xi[None], size=(h, h), mode="bilinear", antialias=True),
                               size=(CROP, CROP), mode="bilinear", align_corners=False)[0]
        if torch.rand(1, generator=g, device="cuda").item() < 0.3:
            qv = int(75 + 25 * torch.rand(1, generator=g, device="cuda").item())
            u8 = ((xi + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
            xi = decode_jpeg(encode_jpeg(u8, quality=qv)).to("cuda").float() / 127.5 - 1
        out.append(xi)
    return torch.stack(out)


def composite(x, kind, vae, m, bf16=False):
    rt = q8(roundtrip(kind, vae, x, bf16=bf16))
    return q8(x * (1 - m) + rt * m)


def loss_fn(logit, m):
    t = F.avg_pool2d(m, 4)
    bce = F.binary_cross_entropy_with_logits(logit, t)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + 0.5 * dice.mean()


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")


@torch.no_grad()
def build_val(vaes, val_ids):
    """fixed validation: per autoencoder 16 blob splices (pixel AUC) and 8 full round trips + 8 untouched crops"""
    g = torch.Generator("cuda").manual_seed(SEED)
    ds = Crops(val_ids, seed=SEED)
    val = {}
    for j, (name, (kind, vae)) in enumerate(vaes.items()):
        x = torch.stack([ds[j * 32 + k] for k in range(32)]).cuda().float() / 127.5 - 1
        m = blobs(16, g, p_none=0, p_full=0)
        xs = composite(x[:16], kind, vae, m)
        xf = q8(roundtrip(kind, vae, x[16:24]))
        val[name] = (xs.half().cpu(), m.half().cpu(), torch.cat([xf, q8(x[24:32])]).half().cpu())
    return val


@torch.no_grad()
def validate(net, val, train_names):
    net.eval()
    rows = {}
    for name, (xs, m, xi) in val.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(net(xs.cuda().float()).float())
            pi = torch.sigmoid(net(xi.cuda().float()).float())
        t = F.avg_pool2d(m.cuda().float(), 4)
        keep = (t < 0.1) | (t > 0.9)                     # ignore the feathered border
        top = pi.flatten(1).topk(pi[0].numel() // 20, 1).values.mean(1).cpu().numpy()
        rows[name] = dict(pixel_auc=auc((t[keep] > 0.5).cpu().numpy(), p[keep].cpu().numpy()),
                          image_auc=auc(np.r_[np.ones(8), np.zeros(8)], top),
                          seen=name in train_names)
    net.train()
    return rows


CACHE_CROP = 416                      # cached crop; training takes random 384 sub-crops of it


@torch.no_grad()
def build_cache(vaes, ids):
    """every training photo cropped once (seeded) and round-tripped once through every autoencoder (70 % in bf16),
    kept in RAM as uint8: the decoders cost ~0.7 s per batch, the network step ~0.03 s"""
    rng = random.Random(SEED + 7)
    crops = []
    for cid in ids:
        im = Image.open(CACHE / f"{cid:012d}.jpg").convert("RGB")
        if min(im.size) < CACHE_CROP:
            s = CACHE_CROP / min(im.size)
            im = im.resize((round(im.width * s) + 1, round(im.height * s) + 1), Image.BICUBIC)
        x0, y0 = rng.randrange(im.width - CACHE_CROP + 1), rng.randrange(im.height - CACHE_CROP + 1)
        crops.append(torch.from_numpy(np.asarray(im.crop((x0, y0, x0 + CACHE_CROP, y0 + CACHE_CROP))).copy())
                     .permute(2, 0, 1))
    orig = torch.stack(crops)
    rts, t0 = {}, time.time()
    for name, (kind, vae) in vaes.items():
        out = torch.empty_like(orig)
        for i in range(0, len(orig), 16):
            x = orig[i:i + 16].cuda().float() / 127.5 - 1
            y = roundtrip(kind, vae, x, bf16=rng.random() < 0.7)
            out[i:i + 16] = ((y + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        rts[name] = out
        print(f"cache {name}: {len(orig)} crops, {time.time() - t0:.0f}s", flush=True)
    return orig, rts


def train(split, steps, batch, lr, vaes, val, orig, rts):
    names = SPLITS[split]
    net = TraceNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator("cuda").manual_seed(SEED + 1)
    rng = np.random.default_rng(SEED + 2)
    out = OUTPUTS / "dtrace"
    out.mkdir(parents=True, exist_ok=True)
    hist, t0, N, J = [], time.time(), len(orig), CACHE_CROP - CROP
    for step in range(1, steps + 1):
        idx = rng.integers(N, size=batch)
        offs = rng.integers(J + 1, size=(batch, 2))
        flip = rng.random(batch) < 0.5
        r_all = rts[names[step % len(names)]]
        xs, rs = [], []
        for i, (oy, ox), f in zip(idx, offs, flip):
            a, b = orig[i, :, oy:oy + CROP, ox:ox + CROP], r_all[i, :, oy:oy + CROP, ox:ox + CROP]
            xs.append(a.flip(-1) if f else a)
            rs.append(b.flip(-1) if f else b)
        x = torch.stack(xs).cuda().float() / 127.5 - 1
        r = torch.stack(rs).cuda().float() / 127.5 - 1
        m = blobs(batch, g)
        xc = augment(q8(x * (1 - m) + r * m), g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(net(xc).float(), m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            v = validate(net, val, names)
            seen = [r_ for r_ in v.values() if r_["seen"]]
            unseen = [r_ for r_ in v.values() if not r_["seen"]]
            line = dict(split=split, step=step, loss=loss.item(), sec=round(time.time() - t0),
                        pix_seen=np.nanmean([r_["pixel_auc"] for r_ in seen]),
                        pix_unseen=np.nanmean([r_["pixel_auc"] for r_ in unseen]) if unseen else None,
                        img_seen=np.nanmean([r_["image_auc"] for r_ in seen]),
                        img_unseen=np.nanmean([r_["image_auc"] for r_ in unseen]) if unseen else None,
                        per_vae={k: (round(r_["pixel_auc"], 3), round(r_["image_auc"], 3)) for k, r_ in v.items()})
            hist.append(line)
            print(json.dumps(line), flush=True)
            torch.save(dict(state_dict=net.state_dict(), split=split, train_vaes=names, steps=step, history=hist),
                       out / f"dtrace_{split}.pt")
    (out / f"dtrace_{split}_history.json").write_text(json.dumps(hist, indent=1))


def train_all(splits, steps, batch, lr, n_photos):
    sp = json.loads((CACHE / "split.json").read_text())
    vaes = {n: load_vae(n) for n in BANK}                # all ten: the cache holds all, each split uses its own
    val = build_val(vaes, sp["val"])
    orig, rts = build_cache(vaes, sp["train"][:n_photos])
    for s in splits:
        train(s, steps, batch, lr, vaes, val, orig, rts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train"])
    ap.add_argument("--splits", nargs="+", default=["A", "B"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--photos", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    prepare() if args.cmd == "prepare" else train_all(args.splits, args.steps, args.batch, args.lr, args.photos)


if __name__ == "__main__":
    main()
