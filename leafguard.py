#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
USAGE = """commands:
  check        verify environment and dataset folders
  prepare      scan, unify labels, cache, dedupe, split
  train        fine-tune a timm model
  evaluate     metrics, calibration, robustness, Grad-CAM
  export       ONNX model, model card, deploy folder
  compare      table of all evaluated runs
  domain-gap   lab-only baseline tested on PlantDoc
  all          prepare, train, evaluate, export
  streamlit run leafguard.py   web app"""

# 1. Config
ROOT_DIR = Path(os.environ.get(
    "LEAFGUARD_ROOT",
    str(Path.home() / "Documents" / "AI Engineering" / "plant disease project"))).expanduser()
WORK_DIR = Path(os.environ.get("LEAFGUARD_WORK", str(ROOT_DIR / "leafguard_output"))).expanduser()
DEPLOY_DIR = Path(os.environ.get("LEAFGUARD_DEPLOY", str(HERE / "deploy"))).expanduser()

PATHS = {
    "plantvillage":   ROOT_DIR / "archive" / "plantvillage dataset" / "color",
    "plantwild":      ROOT_DIR / "archive (1)",
    "fieldplant":     ROOT_DIR / "archive (2)",
    "cassava":        ROOT_DIR / "cassava-leaf-disease-classification",
    "plantdoc":       ROOT_DIR / "PlantDoc-Dataset-master",
    "plantpathology": ROOT_DIR / "plant-pathology-2021-fgvc8",
}
DOMAIN = {"plantvillage": "lab", "plantwild": "field", "fieldplant": "field",
          "cassava": "field", "plantdoc": "field", "plantpathology": "field"}

CACHE_DIR = WORK_DIR / "cache"
RUNS_DIR = WORK_DIR / "runs"
MANIFEST = WORK_DIR / "manifest.csv"
OVERRIDES = WORK_DIR / "label_overrides.csv"

SEED = 42
CACHE_MAX = 384
CACHE_QUALITY = 92
MIN_IMAGES_PER_CLASS = 40
NEAR_DUP_BITS = 10
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# 2. Utilities
def log(*a):
    print(*a, flush=True)


def progress(it, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc, ncols=100, mininterval=1.0)
    except Exception:
        return _plain_progress(it, total, desc)


def _plain_progress(it, total, desc):
    step = max(1, (total or 20) // 20)
    for i, x in enumerate(it, 1):
        if i % step == 0:
            log(f"  {desc}: {i}/{total or '?'}")
        yield x


def read_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def update_json(p, new: dict):
    d = read_json(p, {}) or {}
    d.update(new)
    write_json(p, d)


def set_seed(s):
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def split_key(key):
    crop, _, cond = key.partition("__")
    return crop, cond.replace("_", " ")


def pretty_label(key):
    crop, cond = split_key(key)
    return f"{crop.capitalize()}: {cond}"


# 3. Label unification
_CROPS = {
    "apple": ["apple"], "banana": ["banana"], "basil": ["basil"], "bean": ["bean"],
    "pepper": ["bell pepper", "pepper bell", "pepper", "capsicum"], "blueberry": ["blueberry"],
    "broccoli": ["broccoli"], "cabbage": ["cabbage"], "carrot": ["carrot"], "cassava": ["cassava"],
    "cauliflower": ["cauliflower"], "celery": ["celery"], "cherry": ["cherry"],
    "citrus": ["citrus", "orange", "lemon"], "coffee": ["coffee"], "corn": ["corn", "maize"],
    "cucumber": ["cucumber"], "eggplant": ["eggplant", "brinjal"], "garlic": ["garlic"],
    "ginger": ["ginger"], "grape": ["grapevine", "grape"], "lettuce": ["lettuce"], "maple": ["maple"],
    "peach": ["peach"], "plum": ["plum"], "potato": ["potato"], "raspberry": ["raspberry"],
    "rice": ["rice"], "soybean": ["soybean", "soyabean"], "squash": ["squash"],
    "strawberry": ["strawberry"], "tobacco": ["tobacco"], "tomato": ["tomato"], "zucchini": ["zucchini"],
    "wheat": ["wheat"], "watermelon": ["watermelon"], "pumpkin": ["pumpkin"], "peanut": ["peanut"],
    "onion": ["onion"], "spinach": ["spinach"], "sugarcane": ["sugarcane"], "mango": ["mango"],
    "papaya": ["papaya"], "pomegranate": ["pomegranate"], "cotton": ["cotton"], "tea": ["tea"],
    "okra": ["okra"], "chili": ["chili", "chilli"], "guava": ["guava"], "jackfruit": ["jackfruit"],
}
_CROP_LOOKUP = sorted(((ph, c) for c, phs in _CROPS.items() for ph in phs), key=lambda t: -len(t[0]))

_PROTECT = ["leaf spot", "leaf blight", "leaf mold", "leaf mould", "leaf curl", "leaf scorch",
            "leaf rust", "leaf roll", "leaf miner", "leaf drop"]

_RULES = [
    ("corn", r"gray leaf spot|grey leaf spot", "gray leaf spot"),
    ("corn", r"northern leaf blight|^leaf blight$", "northern leaf blight"),
    ("corn", r"\brust\b", "common rust"),
    ("apple", r"\brust\b", "rust"),
    ("apple", r"\bscab\b", "scab"),
    ("apple", r"frog ?eye", "frog eye leaf spot"),
    ("pepper", r"^(bacterial spot|leaf spot|bacterial leaf spot)$", "bacterial spot"),
    ("tomato", r"^(bacterial spot|bacterial leaf spot)$", "bacterial spot"),
    ("tomato", r"septoria", "septoria leaf spot"),
    ("tomato", r"\bleaf mold\b|\bmold\b|\bmould\b", "leaf mold"),
    ("tomato", r"yellow leaf curl|yellow virus", "yellow leaf curl virus"),
    ("tomato", r"^(tomato )?mosaic virus$", "mosaic virus"),
    ("tomato", r"spider mite", "spider mites"),
    ("grape", r"esca", "esca (black measles)"),
    ("grape", r"isariopsis|^leaf blight|^leaf spot$", "leaf spot"),
    ("citrus", r"greening|haunglongbing|huanglongbing|\bhlb\b", "citrus greening"),
    ("cassava", r"bacterial blight", "bacterial blight"),
    ("cassava", r"brown streak", "brown streak disease"),
    ("cassava", r"green mottle", "green mottle"),
    ("cassava", r"^mosaic", "mosaic disease"),
    ("*", r"^bacterial leaf spot$", "bacterial spot"),
]


def _clean(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_crop(text: str):
    for phrase, canon in _CROP_LOOKUP:
        if text == phrase or text.startswith(phrase + " "):
            return canon, text[len(phrase):].strip()
    return None, text


def _strip_leading_crop(cond: str, crop: str) -> str:
    for ph in _CROPS.get(crop, [crop]):
        if cond.startswith(ph + " "):
            return cond[len(ph) + 1:].strip()
    return cond


def canon_label(raw: str, default_crop: str | None = None):
    flag = ""
    if "___" in raw:
        crop_part, cond_part = raw.split("___", 1)
        crop, _ = _match_crop(_clean(crop_part))
        cond = _clean(cond_part)
        if crop is None:
            crop = (_clean(crop_part).split() or ["unknown"])[0]
            flag = "unknown_crop"
    else:
        txt = _clean(raw)
        crop, cond = _match_crop(txt)
        if crop is None:
            if default_crop:
                crop, cond = default_crop, txt
            else:
                toks = txt.split()
                crop, cond = (toks[0] if toks else "unknown"), " ".join(toks[1:])
                flag = "unknown_crop"
    cond = _strip_leading_crop(cond, crop)
    prot = cond
    for ph in _PROTECT:
        prot = prot.replace(ph, ph.replace(" ", "~"))
    toks = [t for t in prot.split() if t not in ("leaf", "leaves")]
    cond = " ".join(toks).replace("~", " ").strip()
    if cond in ("", "healthy", "normal") or cond.startswith("healthy ") or cond.endswith(" healthy"):
        cond = "healthy"
    else:
        for rc, pat, canon in _RULES:
            if (rc == "*" or rc == crop) and re.search(pat, cond):
                cond = canon
                break
    key = f"{crop}__" + re.sub(r"[^a-z0-9]+", "_", cond).strip("_")
    return crop, cond, key, flag


# 4. Dataset scanners
def _valid(kind: str, p: Path) -> bool:
    try:
        if not p.is_dir():
            return False
        if kind == "plantvillage":
            return sum(1 for d in p.iterdir() if d.is_dir()) >= 30
        if kind == "plantwild":
            return all((p / s).is_dir() for s in ("train", "val", "test"))
        if kind == "fieldplant":
            return (p / "data.yaml").is_file() and (p / "train" / "images").is_dir()
        if kind == "cassava":
            return (p / "train.csv").is_file() and (p / "label_num_to_disease_map.json").is_file()
        if kind == "plantdoc":
            return (p / "train").is_dir() and (p / "test").is_dir() and not (p / "val").exists()
        if kind == "plantpathology":
            return ((p / "train.csv").is_file() and (p / "train_images").is_dir()
                    and not (p / "label_num_to_disease_map.json").exists())
    except OSError:
        return False
    return False


def discover_paths() -> dict:
    found, missing = {}, []
    for kind, p in PATHS.items():
        if _valid(kind, Path(p)):
            found[kind] = Path(p)
        else:
            missing.append(kind)
    if missing and ROOT_DIR.is_dir():
        base_depth = len(ROOT_DIR.parts)
        for cur, dirs, _ in os.walk(ROOT_DIR):
            cp = Path(cur)
            if WORK_DIR in cp.parents or cp == WORK_DIR:
                dirs[:] = []
                continue
            for kind in list(missing):
                if _valid(kind, cp):
                    found[kind] = cp
                    missing.remove(kind)
            if len(cp.parts) - base_depth >= 4:
                dirs[:] = []
    return {"found": found, "missing": missing}


def _images_in(d: Path):
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def _rec(src, raw, path, split_hint=None, group=None, bbox=None, default_crop=None):
    return {"src": src, "raw": raw, "path": str(path), "split_hint": split_hint,
            "group": group, "bbox": bbox, "default_crop": default_crop, "domain": DOMAIN[src]}


def scan_folder_per_class(root: Path, src: str, split_hint=None, default_crop=None):
    out = []
    for cdir in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in _images_in(cdir):
            out.append(_rec(src, cdir.name, f, split_hint, None, None, default_crop))
    return out


def scan_plantvillage(root):
    return scan_folder_per_class(root, "plantvillage")


def scan_plantwild(root):
    out = []
    for sp in ("train", "val", "test"):
        if (root / sp).is_dir():
            out += scan_folder_per_class(root / sp, "plantwild", split_hint=sp)
    return out


def scan_plantdoc(root):
    out = scan_folder_per_class(root / "train", "plantdoc", split_hint="train")
    out += scan_folder_per_class(root / "test", "plantdoc", split_hint="ext_test")
    return out


def scan_fieldplant(root):
    import yaml
    with open(root / "data.yaml", "r", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    out, no_label = [], 0
    for sp in ("train", "valid", "val", "test"):
        idir, ldir = root / sp / "images", root / sp / "labels"
        if not idir.is_dir():
            continue
        for img in _images_in(idir):
            lf = ldir / (img.stem + ".txt")
            if not lf.is_file():
                no_label += 1
                continue
            group = "fieldplant:" + re.split(r"\.rf\.", img.name)[0]
            for line in lf.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    cls = int(float(p[0]))
                    v = [float(x) for x in p[1:]]
                except ValueError:
                    continue
                if len(v) == 4:
                    cx, cy, w, h = v
                else:
                    xs, ys = v[0::2], v[1::2]
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    w, h = max(xs) - min(xs), max(ys) - min(ys)
                if 0 <= cls < len(names):
                    out.append(_rec("fieldplant", str(names[cls]), img, None, group, (cx, cy, w, h)))
    if no_label:
        log(f"  fieldplant: {no_label} images without label file skipped")
    return out


def scan_cassava(root):
    names = read_json(root / "label_num_to_disease_map.json", {})
    out = []
    with open(root / "train.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = root / "train_images" / row["image_id"]
            raw = names.get(str(row["label"]))
            if raw and p.is_file():
                out.append(_rec("cassava", raw, p, None, None, None, "cassava"))
    return out


def scan_plantpathology(root):
    out, skipped = [], 0
    with open(root / "train.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labs = row["labels"].split()
            p = root / "train_images" / row["image"]
            if len(labs) == 1 and labs[0] != "complex" and p.is_file():
                out.append(_rec("plantpathology", labs[0], p, None, None, None, "apple"))
            else:
                skipped += 1
    log(f"  plantpathology: {skipped} multi-label/complex photos skipped (single-label model)")
    return out


SCANNERS = {"plantvillage": scan_plantvillage, "plantwild": scan_plantwild, "fieldplant": scan_fieldplant,
            "cassava": scan_cassava, "plantdoc": scan_plantdoc, "plantpathology": scan_plantpathology}


# 5. Prepare
def _dhash_hex(im) -> str:
    import numpy as np
    from PIL import Image
    g = im.convert("L").resize((17, 16), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes().hex()


def _cache_worker(task):
    src_path, bbox, out_path = task
    try:
        from PIL import Image, ImageOps
        Image.MAX_IMAGE_PIXELS = None
        out = Path(out_path)
        if out.is_file():
            with Image.open(out) as im:
                im.load()
                return (out_path, _dhash_hex(im), im.size[0], im.size[1], None)
        im = Image.open(src_path)
        if bbox is None:
            try:
                im.draft("RGB", (CACHE_MAX, CACHE_MAX))
            except Exception:
                pass
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
        if bbox is not None:
            W, H = im.size
            cx, cy, w, h = bbox
            bw, bh = w * W, h * H
            x1 = max(0, int(cx * W - bw / 2 - 0.08 * bw))
            y1 = max(0, int(cy * H - bh / 2 - 0.08 * bh))
            x2 = min(W, int(cx * W + bw / 2 + 0.08 * bw))
            y2 = min(H, int(cy * H + bh / 2 + 0.08 * bh))
            if (x2 - x1) < 32 or (y2 - y1) < 32:
                return (out_path, None, 0, 0, "box too small")
            im = im.crop((x1, y1, x2, y2))
        im.thumbnail((CACHE_MAX, CACHE_MAX), Image.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.jpg")
        im.save(tmp, "JPEG", quality=CACHE_QUALITY)
        os.replace(tmp, out)
        return (out_path, _dhash_hex(im), im.size[0], im.size[1], None)
    except Exception as e:
        return (out_path, None, 0, 0, repr(e)[:160])


def _hex_matrix(hexes):
    import numpy as np
    return np.frombuffer(b"".join(bytes.fromhex(h) for h in hexes), dtype=np.uint8).reshape(len(hexes), 32)


def load_overrides():
    ov = {}
    if OVERRIDES.is_file():
        with open(OVERRIDES, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("label_key"):
                    ov[(row["source"], row["raw_label"])] = row["label_key"].strip()
    return ov


def prepare_cmd(a):
    import numpy as np
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.model_selection import StratifiedGroupKFold
    import warnings
    t0 = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    disc = discover_paths()
    if disc["missing"]:
        log("[!] datasets not found:", ", ".join(disc["missing"]), "-> fix PATHS / ROOT_DIR, then rerun")
        if len(disc["found"]) == 0:
            return 1
        log("    continuing with the datasets that were found")
    log("== 1/6 scanning datasets")
    recs = []
    for kind, root in disc["found"].items():
        r = SCANNERS[kind](root)
        log(f"  {kind:15s} {len(r):7d} samples   ({root})")
        recs += r
    if not recs:
        log("[!] nothing found"); return 1

    log("== 2/6 unifying labels")
    ov = load_overrides()
    for r in recs:
        crop, cond, key, flag = canon_label(r["raw"], r["default_crop"])
        if (r["src"], r["raw"]) in ov:
            key, flag = ov[(r["src"], r["raw"])], "override"
        r["crop"], r["cond"], r["label"], r["flag"] = crop, cond, key, flag

    log("== 3/6 caching resized images + hashing (first run takes a few minutes)")
    import hashlib
    tasks = []
    for r in recs:
        h = hashlib.md5(f"{r['path']}|{r['bbox']}".encode()).hexdigest()
        r["id"] = h
        r["cache"] = str(Path("cache") / h[:2] / f"{h}.jpg")
        tasks.append((r["path"], r["bbox"], str(WORK_DIR / r["cache"])))
    nproc = max(1, (os.cpu_count() or 2) - 1)
    results = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for res in progress(ex.map(_cache_worker, tasks, chunksize=64), total=len(tasks), desc="cache"):
            results.append(res)
    good, bad = [], 0
    for r, (_, hx, w, h, err) in zip(recs, results):
        if hx is None:
            bad += 1
            continue
        r["hash"], r["w"], r["h"] = hx, w, h
        good.append(r)
    log(f"  usable: {len(good)}   unreadable/too small: {bad}")
    recs = good

    log("== 4/6 removing duplicates + guarding the PlantDoc test set")
    prio = lambda r: (0 if r["split_hint"] == "ext_test" else 1 if r["split_hint"] in ("test", "val")
                      else 2 if r["split_hint"] == "train" else 3)
    by_hash = defaultdict(list)
    for r in recs:
        by_hash[r["hash"]].append(r)
    keep, n_dup, n_conf = [], 0, 0
    for hx, lst in by_hash.items():
        if len(lst) == 1:
            keep.append(lst[0]); continue
        if len({x["label"] for x in lst}) > 1:
            n_conf += len(lst); continue
        lst.sort(key=prio)
        keep.append(lst[0]); n_dup += len(lst) - 1
    ext = [r for r in keep if r["split_hint"] == "ext_test"]
    oth = [r for r in keep if r["split_hint"] != "ext_test"]
    n_leak = 0
    if ext and oth:
        pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        E, O = _hex_matrix([r["hash"] for r in ext]), _hex_matrix([r["hash"] for r in oth])
        leak = np.zeros(len(oth), dtype=bool)
        for i in range(0, len(E), 8):
            d = pop[np.bitwise_xor(O[None, :, :], E[i:i + 8, None, :])].sum(axis=2, dtype=np.uint16)
            leak |= (d <= NEAR_DUP_BITS).any(axis=0)
        n_leak = int(leak.sum())
        oth = [r for r, l in zip(oth, leak) if not l]
    keep = ext + oth
    log(f"  exact duplicates removed: {n_dup}   conflicting-label duplicates removed: {n_conf}   "
        f"training images too similar to a PlantDoc test photo removed: {n_leak}")

    log("== 5/6 filtering rare classes + building leak-free splits")
    cnt = Counter(r["label"] for r in keep if r["split_hint"] != "ext_test")
    kept_classes = sorted(c for c, n in cnt.items() if n >= max(1, a.min_images))
    dropped = {c: n for c, n in cnt.items() if n < a.min_images}
    cset = set(kept_classes)
    final = []
    for r in keep:
        if r["split_hint"] == "ext_test":
            r["split"] = "ext_test" if r["label"] in cset else "ext_unmapped"
            final.append(r)
        elif r["label"] in cset:
            r["split"] = r["split_hint"]
            final.append(r)
    pool = [r for r in final if r["split"] is None]
    if pool:
        c2i = {c: i for i, c in enumerate(kept_classes)}
        gid, groups = {}, []
        for r in pool:
            g = r["group"] or r["id"]
            groups.append(gid.setdefault(g, len(gid)))
        y = np.array([c2i[r["label"]] for r in pool])
        fold = np.zeros(len(pool), dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, (_, te) in enumerate(sgkf.split(np.zeros(len(pool)), y, np.array(groups))):
                fold[te] = k
        for r, f in zip(pool, fold):
            r["split"] = "test" if f == 0 else "val" if f == 1 else "train"

    log("== 6/6 writing manifest + reports")
    cols = ["id", "cache", "src", "domain", "raw", "label", "crop", "cond", "group", "split", "hash", "w", "h"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in final if r["split"] in
                       ("train", "val", "test", "ext_test", "ext_unmapped")])
    df.to_csv(MANIFEST, index=False)
    write_json(WORK_DIR / "classes.json", {"classes": kept_classes,
                                          "display": [pretty_label(c) for c in kept_classes]})
    tot = Counter((r["src"], r["raw"]) for r in recs)
    lm = {}
    for r in recs:
        lm[(r["src"], r["raw"])] = (r["crop"], r["cond"], r["label"], r["flag"])
    with open(WORK_DIR / "label_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "raw_label", "crop", "condition", "label_key", "n_images", "status"])
        for (s, raw), (cr, co, key, fl) in sorted(lm.items()):
            status = fl or ("kept" if key in cset else "dropped_rare")
            if fl == "override" and key not in cset:
                status = "override_dropped_rare"
            w.writerow([s, raw, cr, co, key, tot[(s, raw)], status])
    if not OVERRIDES.is_file():
        with open(OVERRIDES, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["source", "raw_label", "label_key"])
    rep = {
        "n_images": int(len(df)), "n_classes": len(kept_classes),
        "per_split": df.split.value_counts().to_dict(),
        "per_source_split": {s: g.split.value_counts().to_dict() for s, g in df.groupby("src")},
        "per_domain": df[df.split.isin(["train", "val", "test"])].domain.value_counts().to_dict(),
        "dropped_rare_classes": dropped, "exact_duplicates_removed": n_dup,
        "conflicting_duplicates_removed": n_conf, "near_duplicates_of_plantdoc_test_removed": n_leak,
        "ext_unmapped_images": int((df.split == "ext_unmapped").sum()),
        "unknown_crop_labels": sorted({r["raw"] for r in recs if r["flag"] == "unknown_crop"}),
    }
    write_json(WORK_DIR / "data_report.json", rep)
    _plot_class_distribution(df, WORK_DIR / "class_distribution.png")
    log("\n  images per split:", rep["per_split"])
    log("  images per domain (train+val+test):", rep["per_domain"])
    log(f"  classes: {len(kept_classes)}   (dropped as too rare: {len(dropped)})")
    if rep["unknown_crop_labels"]:
        log("  [review] labels with an unrecognised crop name:", rep["unknown_crop_labels"][:10])
    log(f"  review the class mapping:  {WORK_DIR / 'label_map.csv'}")
    log(f"  fix a mapping by adding a row to: {OVERRIDES}  (columns: source,raw_label,label_key), then rerun prepare")
    log(f"  done in {(time.time() - t0) / 60:.1f} min")
    return 0


def _plot_class_distribution(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.split.isin(["train", "val", "test"])]
    piv = d.groupby(["label", "domain"]).size().unstack(fill_value=0)
    for c in ("lab", "field"):
        if c not in piv:
            piv[c] = 0
    piv = piv.loc[piv.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(piv))))
    ax.barh([pretty_label(k) for k in piv.index], piv["lab"], label="lab photos", color="#8fa8c4")
    ax.barh([pretty_label(k) for k in piv.index], piv["field"], left=piv["lab"], label="field photos",
            color="#2f7d4f")
    ax.set_xlabel("images"); ax.set_title("Images per class"); ax.legend()
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def load_manifest():
    import pandas as pd
    if not MANIFEST.is_file():
        raise SystemExit(f"manifest not found - run:  python {Path(__file__).name} prepare")
    df = pd.read_csv(MANIFEST, keep_default_na=False,
                     dtype={"id": str, "hash": str, "cache": str, "group": str, "raw": str, "label": str})
    df["fpath"] = df["cache"].map(lambda p: str(WORK_DIR / p))
    return df


# 6. Check
def check_cmd(a):
    log(f"python {sys.version.split()[0]}   root: {ROOT_DIR}")
    for m in ["torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "scipy", "PIL", "matplotlib",
              "yaml", "onnx", "onnxruntime", "streamlit", "tqdm"]:
        try:
            mod = __import__(m)
            log(f"  OK       {m:12s} {getattr(mod, '__version__', '')}")
        except Exception:
            log(f"  MISSING  {m}")
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            log(f"  GPU: {p.name}  {p.total_memory / 2**30:.1f} GB   bf16: {torch.cuda.is_bf16_supported()}")
        else:
            log("  [!] CUDA not available - training would run on CPU (very slow)")
    except Exception:
        pass
    disc = discover_paths()
    for k in PATHS:
        log(f"  dataset {k:15s} {'FOUND  ' + str(disc['found'][k]) if k in disc['found'] else 'NOT FOUND'}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(WORK_DIR).free / 2**30
    log(f"  free disk at {WORK_DIR}: {free:.0f} GB (cache needs ~8 GB)")
    log("  manifest:", "present" if MANIFEST.is_file() else "not built yet (run prepare)")
    return 0


# 7. Torch building blocks
def get_transforms(size, mean, std, train):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode as IM
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.75, 1.3333), interpolation=IM.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(0.2),
            T.TrivialAugmentWide(interpolation=IM.BILINEAR),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.03)], p=0.5),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.12), value="random"),
        ])
    return T.Compose([T.Resize(int(round(size * 1.14)), interpolation=IM.BICUBIC), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(mean, std)])


_DS = None


def dataset_class():
    global _DS
    if _DS is None:
        from PIL import Image
        from torch.utils.data import Dataset

        class PlantDS(Dataset):
            def __init__(self, paths, labels, tfm, corrupt=None):
                self.paths, self.labels, self.tfm, self.corrupt = list(paths), list(labels), tfm, corrupt

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, i):
                with Image.open(self.paths[i]) as im:
                    im = im.convert("RGB")
                if self.corrupt is not None:
                    im = self.corrupt(im)
                return self.tfm(im), int(self.labels[i])
        _DS = PlantDS
    return _DS


def default_workers():
    return max(2, min(12, (os.cpu_count() or 4) - 2))


def make_loader(paths, labels, tfm, bs, workers, sampler=None, drop_last=False, corrupt=None):
    from torch.utils.data import DataLoader
    ds = dataset_class()(paths, labels, tfm, corrupt)
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=True, drop_last=drop_last,
              sampler=sampler, shuffle=False)
    if workers > 0:
        kw.update(persistent_workers=sampler is not None, prefetch_factor=4)
    return DataLoader(ds, **kw)


def build_model(arch, num_classes, pretrained=True, drop_rate=0.2, drop_path=0.1):
    import timm
    kw = dict(pretrained=pretrained, num_classes=num_classes)
    try:
        return timm.create_model(arch, drop_rate=drop_rate, drop_path_rate=drop_path, **kw)
    except TypeError:
        return timm.create_model(arch, **kw)
    except Exception as e:
        if pretrained:
            raise SystemExit(f"[!] could not load pretrained weights for '{arch}': {e}\n"
                             "    timm downloads them once (internet needed). Check the connection and retry.")
        raise


def amp_setup(device):
    import torch
    use = device.type == "cuda"
    dt = torch.bfloat16 if (use and torch.cuda.is_bf16_supported()) else torch.float16
    return use, dt


class EMA:

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n, self._pairs = decay, 0, None

    def update(self, model):
        import torch
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        if self._pairs is None:
            msd, esd = model.state_dict(), self.module.state_dict()
            fl = [k for k in esd if esd[k].dtype.is_floating_point]
            self._pairs = ([esd[k] for k in fl], [msd[k] for k in fl],
                           [(esd[k], msd[k]) for k in esd if not esd[k].dtype.is_floating_point])
        ema_t, mod_t, other = self._pairs
        with torch.no_grad():
            torch._foreach_mul_(ema_t, d)
            torch._foreach_add_(ema_t, mod_t, alpha=1 - d)
            for e, m in other:
                e.copy_(m)


def param_groups(model, lr, wd, head_mult):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = []
    for is_head in (False, True):
        for decay in (True, False):
            ps = [p for p in model.parameters()
                  if p.requires_grad and (id(p) in head_ids) == is_head and (p.ndim > 1) == decay]
            if ps:
                groups.append({"params": ps, "lr": lr * (head_mult if is_head else 1.0),
                               "weight_decay": wd if decay else 0.0})
    return groups


def predict_logits(model, loader, device, use_amp, amp_dtype, tta=False):
    import torch
    model.eval()
    outs = []
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                lo = model(x)
                if tta:
                    lo = (lo + model(torch.flip(x, dims=[3]))) / 2
            outs.append(lo.float().cpu())
    return torch.cat(outs).numpy()


# 8. Train
def train_cmd(a):
    import numpy as np
    import torch
    import torch.nn as nn
    import timm
    from torch.utils.data import WeightedRandomSampler
    from sklearn.metrics import f1_score
    set_seed(a.seed)
    df = load_manifest()
    sources = None if (not a.sources or a.sources == "all") else [s.strip() for s in a.sources.split(",")]
    base = df[df.split.isin(["train", "val", "test"])]
    if sources:
        base = base[base.src.isin(sources)]
    classes = sorted(base[base.split == "train"].label.unique())
    if len(classes) < 2:
        raise SystemExit("[!] fewer than 2 classes in the training split - run prepare first")
    cset = set(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    tr = base[(base.split == "train") & base.label.isin(cset)].reset_index(drop=True)
    va = base[(base.split == "val") & base.label.isin(cset)].reset_index(drop=True)
    ext = df[(df.split == "ext_test") & df.label.isin(cset)].reset_index(drop=True)
    run_name = a.run_name or a.arch
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dt = amp_setup(device)
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log(f"== run '{run_name}'  arch={a.arch}  classes={len(classes)}  train={len(tr)}  val={len(va)}  "
        f"external(PlantDoc test)={len(ext)}")
    log(f"   device={device}  amp={'bf16' if amp_dt == torch.bfloat16 else 'fp16'}  img={a.img_size}  bs={a.batch_size}")

    model = build_model(a.arch, len(classes), not a.no_pretrained, a.drop, a.drop_path)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = [float(x) for x in cfg["mean"]], [float(x) for x in cfg["std"]]
    model.to(device).to(memory_format=torch.channels_last)
    ema = EMA(model, a.ema_decay)

    tf_tr = get_transforms(a.img_size, mean, std, True)
    tf_ev = get_transforms(a.img_size, mean, std, False)
    y_tr = tr.label.map(c2i).to_numpy()
    cnt = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
    w = cnt[y_tr] ** -0.5 * np.where(tr.domain.to_numpy() == "field", a.field_boost, 1.0)
    n_samp = a.samples_per_epoch if a.samples_per_epoch > 0 else len(tr)
    gen = torch.Generator().manual_seed(a.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=n_samp, replacement=True, generator=gen)
    nw = a.workers or default_workers()
    tl = make_loader(tr.fpath, y_tr, tf_tr, a.batch_size, nw, sampler=sampler, drop_last=True)
    y_va = va.label.map(c2i).to_numpy()
    vl = make_loader(va.fpath, y_va, tf_ev, a.batch_size * 2, nw)
    y_ex = ext.label.map(c2i).to_numpy()
    el = make_loader(ext.fpath, y_ex, tf_ev, a.batch_size * 2, min(nw, 4)) if len(ext) else None
    steps = len(tl)

    opt = torch.optim.AdamW(param_groups(model, a.lr, a.wd, a.head_mult), betas=(0.9, 0.999))
    total, warm = a.epochs * steps, max(1, int(a.warmup_epochs * steps))

    def lam(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dt == torch.float16))
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    hist, best_f1, bad, start_ep = [], -1.0, 0, 0
    last = run_dir / "last.pt"
    if a.resume and last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); ema.n = ck["ema_n"]
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, best_f1, bad, start_ep = ck["hist"], ck["best_f1"], ck["bad"], ck["epoch"] + 1
        log(f"   resumed from epoch {start_ep}")

    meta = {"arch": a.arch, "classes": classes, "mean": mean, "std": std, "img_size": a.img_size,
            "sources": sources, "args": {k: v for k, v in vars(a).items() if not callable(v)}}
    write_json(run_dir / "run_info.json", {**{k: v for k, v in meta.items() if k != "classes"},
                                           "n_classes": len(classes), "n_train": len(tr), "n_val": len(va),
                                           "n_external": len(ext)})
    t_start = time.time()
    try:
        for ep in range(start_ep, a.epochs):
            model.train()
            t0, seen = time.time(), 0
            sum_loss = torch.zeros((), device=device)
            sum_ok = torch.zeros((), device=device)
            for x, y in progress(tl, total=steps, desc=f"epoch {ep + 1}/{a.epochs}"):
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dt, enabled=use_amp):
                    out = model(x)
                    loss = crit(out.float(), y)
                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                ema.update(model)
                seen += x.size(0)
                sum_loss += loss.detach().float() * x.size(0)
                sum_ok += (out.argmax(1) == y).sum()
            tl_sum, tc = float(sum_loss), int(sum_ok)
            train_time = time.time() - t0
            vlog = predict_logits(ema.module, vl, device, use_amp, amp_dt)
            pred = vlog.argmax(1)
            v_acc = float((pred == y_va).mean())
            v_f1 = float(f1_score(y_va, pred, average="macro", labels=np.unique(y_va), zero_division=0))
            v_loss = float(nll_np(vlog, y_va, 1.0))
            e_acc = float("nan")
            if el is not None:
                elog = predict_logits(ema.module, el, device, use_amp, amp_dt)
                e_acc = float((elog.argmax(1) == y_ex).mean())
            row = {"epoch": ep + 1, "train_loss": tl_sum / max(1, seen), "train_acc": tc / max(1, seen),
                   "val_loss": v_loss, "val_acc": v_acc, "val_macro_f1": v_f1, "plantdoc_acc": e_acc,
                   "lr": opt.param_groups[0]["lr"], "img_per_s": seen / train_time,
                   "minutes": (time.time() - t_start) / 60}
            hist.append(row)
            log(f"   ep {ep + 1:02d}/{a.epochs}  loss {row['train_loss']:.3f}  val loss {v_loss:.3f}  "
                f"acc {v_acc:.4f}  macroF1 {v_f1:.4f}  PlantDoc acc {e_acc:.4f}  "
                f"{row['img_per_s']:.0f} img/s  {row['minutes']:.1f} min")
            if v_f1 > best_f1:
                best_f1, bad = v_f1, 0
                torch.save({**meta, "state_dict": ema.module.state_dict(), "epoch": ep + 1, "val_f1": v_f1},
                           run_dir / "best.pt")
                log("   * new best saved")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "ema_n": ema.n,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
                        "best_f1": best_f1, "bad": bad, "epoch": ep}, last)
            _write_history(run_dir, hist)
            if bad >= a.patience:
                log(f"   early stop: no val macro-F1 gain for {a.patience} epochs")
                break
    except KeyboardInterrupt:
        log("\n   interrupted. Continue later with the same command plus --resume")
    _write_history(run_dir, hist)
    _plot_history(hist, run_dir / "training_curves.png")
    log(f"== training done. best val macro-F1 {best_f1:.4f}   weights: {run_dir / 'best.pt'}")
    log(f"   next:  python {Path(__file__).name} evaluate --run {run_name}")
    return 0


def _write_history(run_dir, hist):
    if not hist:
        return
    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)


def _plot_history(hist, path):
    if not hist:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train"); ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val accuracy")
    ax[1].plot(ep, [h["val_macro_f1"] for h in hist], label="val macro-F1")
    ax[1].set_title("validation"); ax[1].legend()
    ax[2].plot(ep, [h["plantdoc_acc"] for h in hist], color="#2f7d4f")
    ax[2].set_title("PlantDoc (real-world) accuracy - monitored, never used to pick the model")
    ax[2].title.set_fontsize(9)
    for x in ax:
        x.set_xlabel("epoch"); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# 9. Metrics
def softmax_np(z, T=1.0):
    import numpy as np
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_np(logits, y, T=1.0):
    import numpy as np
    p = softmax_np(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits, y):
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t: nll_np(logits, y, math.exp(t)), bounds=(-2.0, 3.0), method="bounded")
    return float(math.exp(res.x))


def ece_np(probs, y, n_bins=15):
    import numpy as np
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    ok = pred == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
            rows.append((float(lo), float(hi), float(ok[m].mean()), float(conf[m].mean()), int(m.sum())))
    return float(ece), rows


def brier_np(probs, y):
    import numpy as np
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def topk_acc(logits, y, k):
    import numpy as np
    k = min(k, logits.shape[1])
    top = np.argsort(-logits, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def compute_metrics(y, logits, T=1.0):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
    probs = softmax_np(logits, T)
    pred = probs.argmax(1)
    present = np.unique(y)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=present, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y, pred, labels=present, average="weighted", zero_division=0)
    aucs, aps = [], []
    for c in present:
        pos = y == c
        if pos.all() or not pos.any():
            continue
        aucs.append(roc_auc_score(pos, probs[:, c]))
        aps.append(average_precision_score(pos, probs[:, c]))
    ece, _ = ece_np(probs, y)
    return {"n": int(len(y)), "n_classes_present": int(len(present)), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(r), "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f), "weighted_f1": float(wf), "top3_accuracy": topk_acc(logits, y, 3),
            "top5_accuracy": topk_acc(logits, y, 5), "roc_auc_macro_ovr": float(np.mean(aucs)) if aucs else None,
            "mAP_macro": float(np.mean(aps)) if aps else None, "nll": nll_np(logits, y, T),
            "brier": brier_np(probs, y), "ece": ece}


def per_class_table(y, logits, classes, T=1.0):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    probs = softmax_np(logits, T)
    pred = probs.argmax(1)
    C = len(classes)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(C)), zero_division=0)
    rows = []
    for c in range(C):
        auc = None
        pos = y == c
        if pos.any() and not pos.all():
            auc = float(roc_auc_score(pos, probs[:, c]))
        rows.append({"class": classes[c], "name": pretty_label(classes[c]), "support": int(s[c]),
                     "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "auc": auc})
    return rows


def pick_conf_threshold(conf, correct, target):
    import numpy as np
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order].astype(float)
    acc = np.cumsum(ok) / (np.arange(len(ok)) + 1)
    idx = np.where(acc >= target)[0]
    if len(idx) == 0:
        return 0.5, 0.0
    i = int(idx.max())
    return float(c[i]), float((i + 1) / len(ok))


def risk_coverage(conf, correct):
    import numpy as np
    order = np.argsort(-conf)
    ok = correct[order].astype(float)
    n = np.arange(len(ok)) + 1
    return n / len(ok), np.cumsum(ok) / n


# 10. Evaluate
def _latest_run():
    runs = sorted((p for p in RUNS_DIR.glob("*/best.pt")), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("[!] no trained run found - run train first")
    return runs[-1].parent.name


def load_run(run_name):
    import torch
    run_dir = RUNS_DIR / run_name
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(f"[!] {run_dir / 'best.pt'} not found")
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return ck, model, run_dir


def _corruptions():
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    def noise(sig):
        def f(im):
            a = np.asarray(im).astype(np.float32)
            a = a + np.random.default_rng(0).normal(0, sig, a.shape)
            return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return f

    def jpeg(q):
        def f(im):
            b = io.BytesIO()
            im.save(b, "JPEG", quality=q)
            b.seek(0)
            return Image.open(b).convert("RGB")
        return f

    def lowres(im):
        w, h = im.size
        return im.resize((max(8, w // 5), max(8, h // 5))).resize((w, h))

    def occlude(im):
        w, h = im.size
        bw, bh = int(w * 0.5), int(h * 0.5)
        im = im.copy()
        im.paste((0, 0, 0), (w // 4, h // 4, w // 4 + bw, h // 4 + bh))
        return im

    return {"blur (r=2)": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
            "blur (r=5)": lambda im: im.filter(ImageFilter.GaussianBlur(5)),
            "dark (x0.4)": lambda im: ImageEnhance.Brightness(im).enhance(0.4),
            "bright (x1.8)": lambda im: ImageEnhance.Brightness(im).enhance(1.8),
            "low contrast": lambda im: ImageEnhance.Contrast(im).enhance(0.4),
            "noise (s=15)": noise(15), "noise (s=35)": noise(35),
            "rotate 30": lambda im: im.rotate(30, fillcolor=(0, 0, 0)),
            "JPEG q=15": jpeg(15), "low resolution": lowres, "occlusion 25%": occlude}


def _target_layer(model):
    for name in ("stages", "layer4", "blocks", "features"):
        m = getattr(model, name, None)
        if m is not None:
            if name == "layer4":
                return m
            try:
                return m[-1]
            except Exception:
                return m
    return None


def gradcam_batch(model, x):
    import torch
    import torch.nn.functional as F
    layer = _target_layer(model)
    if layer is None:
        raise RuntimeError("no target layer")
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logits = model(x)
            cls = logits.argmax(1)
            score = logits.gather(1, cls[:, None]).sum()
            grads = torch.autograd.grad(score, store["a"])[0]
    finally:
        h.remove()
    act = store["a"].detach()
    wts = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((wts * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    mn = cam.amin(dim=(1, 2), keepdim=True)
    mx = cam.amax(dim=(1, 2), keepdim=True)
    cam = (cam - mn) / (mx - mn + 1e-8)
    return cam.cpu().numpy(), cls.cpu().numpy()


def bench_torch(model, device, size, use_amp, amp_dt):
    import torch
    res = {"params_million": sum(p.numel() for p in model.parameters()) / 1e6}
    res["fp32_size_mb"] = res["params_million"] * 4
    if device.type == "cuda":
        x = torch.randn(64, 3, size, size, device=device).to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dt, enabled=use_amp):
            for _ in range(3):
                model(x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10):
                model(x)
            torch.cuda.synchronize()
            res["gpu_images_per_s_bs64"] = 640 / (time.time() - t)
            x1 = x[:1]
            torch.cuda.synchronize(); t = time.time()
            for _ in range(30):
                model(x1)
            torch.cuda.synchronize()
            res["gpu_latency_ms_bs1"] = (time.time() - t) / 30 * 1000
    m = copy.deepcopy(model).cpu().float().to(memory_format=torch.contiguous_format).eval()
    torch.set_num_threads(4)
    x = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        for _ in range(3):
            m(x)
        t = time.time()
        for _ in range(15):
            m(x)
    res["cpu_latency_ms_bs1_4threads"] = (time.time() - t) / 15 * 1000
    return res


def evaluate_cmd(a):
    import numpy as np
    import torch
    run = a.run or _latest_run()
    ck, model, run_dir = load_run(run)
    classes = ck["classes"]
    C = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifest()
    base = df[df.split.isin(["train", "val", "test"])]
    if ck.get("sources"):
        base = base[base.src.isin(ck["sources"])]
    cset = set(classes)
    frames = {"val": base[(base.split == "val") & base.label.isin(cset)],
              "test": base[(base.split == "test") & base.label.isin(cset)],
              "plantdoc_test": df[(df.split == "ext_test") & df.label.isin(cset)]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dt = amp_setup(device)
    model.to(device).to(memory_format=torch.channels_last)
    tf = get_transforms(ck["img_size"], ck["mean"], ck["std"], False)
    nw, bs = default_workers(), 128
    log(f"== evaluating run '{run}'  (classes={C}, img={ck['img_size']}, TTA={'on' if a.tta else 'off'})")
    logits, ys = {}, {}
    for name in list(frames):
        d = frames[name].reset_index(drop=True)
        if len(d) == 0:
            frames.pop(name)
            continue
        frames[name] = d
        ys[name] = d.label.map(c2i).to_numpy()
        logits[name] = predict_logits(model, make_loader(d.fpath, ys[name], tf, bs, nw), device, use_amp, amp_dt, a.tta)
        log(f"   {name:14s} n={len(d):6d}  accuracy={(logits[name].argmax(1) == ys[name]).mean():.4f}")
    np.savez_compressed(out / "logits.npz", **{f"logits_{k}": v for k, v in logits.items()},
                        **{f"y_{k}": v for k, v in ys.items()})
    M = {"run": run, "arch": ck["arch"], "n_classes": C, "img_size": ck["img_size"], "tta": bool(a.tta)}

    # calibration
    cal = {"temperature": 1.0, "conf_threshold": 0.5, "maxlogit_threshold": float("-inf")}
    if "val" in logits:
        vm = frames["val"].domain.to_numpy() == "field"
        if vm.sum() < 200:
            vm[:] = True
        lv, yv = logits["val"][vm], ys["val"][vm]
        T = fit_temperature(lv, yv)
        pv = softmax_np(lv, T)
        thr, cov = pick_conf_threshold(pv.max(1), pv.argmax(1) == yv, a.target_acc)
        cal = {"temperature": T, "conf_threshold": thr, "target_accuracy": a.target_acc,
               "val_coverage_at_threshold": cov, "maxlogit_threshold": float(np.percentile(lv.max(1), 2)),
               "calibrated_on": f"{int(vm.sum())} validation photos ({'field' if vm.sum() < len(vm) else 'all'})"}
    else:
        T = 1.0
    write_json(out / "calibration.json", cal)
    M["calibration"] = cal

    # metrics per set
    for name in logits:
        m_raw = compute_metrics(ys[name], logits[name], 1.0)
        m_cal = compute_metrics(ys[name], logits[name], T)
        pr = softmax_np(logits[name], T)
        ok = pr.argmax(1) == ys[name]
        sel = pr.max(1) >= cal["conf_threshold"]
        M[name] = {**m_raw, "ece_after_temperature": m_cal["ece"], "nll_after_temperature": m_cal["nll"],
                   "brier_after_temperature": m_cal["brier"],
                   "selective": {"threshold": cal["conf_threshold"], "coverage": float(sel.mean()),
                                 "accuracy_when_answered": float(ok[sel].mean()) if sel.any() else None}}
        log(f"   {name:14s} acc {m_raw['accuracy']:.4f}  macroF1 {m_raw['macro_f1']:.4f}  top3 {m_raw['top3_accuracy']:.4f}  "
            f"ECE {m_raw['ece']:.4f} -> {m_cal['ece']:.4f} (T={T:.2f})")

    # breakdowns
    if "test" in frames:
        d, y, pred = frames["test"], ys["test"], logits["test"].argmax(1)
        d = d.assign(ok=(pred == y))
        M["test_by_source"] = {s: {"n": int(len(g)), "accuracy": float(g.ok.mean())} for s, g in d.groupby("src")}
        M["test_by_domain"] = {s: {"n": int(len(g)), "accuracy": float(g.ok.mean())} for s, g in d.groupby("domain")}
        crop = d.groupby("crop").ok.agg(["mean", "size"]).sort_values("mean")
        M["test_by_crop"] = {k: {"n": int(r["size"]), "accuracy": float(r["mean"])} for k, r in crop.iterrows()}

    # per-class tables and confusion
    from sklearn.metrics import confusion_matrix
    for name in [n for n in ("test", "plantdoc_test") if n in logits]:
        rows = per_class_table(ys[name], logits[name], classes, T)
        with open(out / f"per_class_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        pred = logits[name].argmax(1)
        idx = sorted(set(ys[name].tolist()) | set(pred.tolist()))
        cm = confusion_matrix(ys[name], pred, labels=idx)
        _plot_confusion(cm, [pretty_label(classes[i]) for i in idx], out / f"confusion_{name}.png",
                        f"Confusion matrix - {name} (row-normalised)")
        pairs = [(int(cm[i, j]), classes[idx[i]], classes[idx[j]]) for i in range(len(idx))
                 for j in range(len(idx)) if i != j and cm[i, j] > 0]
        pairs.sort(reverse=True)
        with open(out / f"top_confusions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["count", "true_class", "predicted_class"])
            for c, t, p in pairs[:40]:
                w.writerow([c, t, p])
        _plot_per_class_f1(rows, out / f"worst_classes_{name}.png", name)
        _plot_roc(ys[name], softmax_np(logits[name], T), C, out / f"roc_pr_{name}.png", name)

    # calibration and selective-prediction plots
    sets = [n for n in ("test", "plantdoc_test") if n in logits]
    if sets:
        _plot_reliability({n: (softmax_np(logits[n], 1.0), softmax_np(logits[n], T), ys[n]) for n in sets},
                          out / "reliability.png", T)
        _plot_risk_coverage({n: (softmax_np(logits[n], T).max(1),
                                 softmax_np(logits[n], T).argmax(1) == ys[n]) for n in sets},
                            cal["conf_threshold"], out / "risk_coverage.png")

    # error grid and Grad-CAM
    ref = "plantdoc_test" if ("plantdoc_test" in frames and len(frames["plantdoc_test"]) >= 8) else "test"
    if ref in frames:
        try:
            _error_and_gradcam(model, frames[ref], ys[ref], logits[ref], classes, tf, ck, device, T,
                               out, ref)
        except Exception as e:
            log(f"   [skip] Grad-CAM/error grid: {e}")

    # robustness
    if not a.no_robustness:
        rng = np.random.default_rng(SEED)
        parts = [frames[n] for n in ("test", "plantdoc_test") if n in frames]
        import pandas as pd
        pool = pd.concat(parts, ignore_index=True)
        take = rng.choice(len(pool), size=min(a.robust_n, len(pool)), replace=False)
        sub = pool.iloc[take].reset_index(drop=True)
        ysub = sub.label.map(c2i).to_numpy()
        res = {}
        log(f"   robustness on {len(sub)} images x {len(_corruptions()) + 1} conditions")
        lg = predict_logits(model, make_loader(sub.fpath, ysub, tf, bs, nw), device, use_amp, amp_dt)
        res["clean"] = float((lg.argmax(1) == ysub).mean())
        for k, fn in _corruptions().items():
            lg = predict_logits(model, make_loader(sub.fpath, ysub, tf, bs, nw, corrupt=fn), device, use_amp, amp_dt)
            res[k] = float((lg.argmax(1) == ysub).mean())
        M["robustness_accuracy"] = res
        _plot_robustness(res, out / "robustness.png")

    # speed and size
    try:
        M["speed"] = bench_torch(model, device, ck["img_size"], use_amp, amp_dt)
    except Exception as e:
        log(f"   [skip] speed benchmark: {e}")
    write_json(out / "metrics.json", M)
    _write_report(M, out / "report.md")
    log(f"== done. Everything is in: {out}\n   start with report.md")
    log(f"   next:  python {Path(__file__).name} export --run {run}")
    return 0


# plots
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_confusion(cm, labels, path, title):
    import numpy as np
    plt = _mpl()
    n = len(labels)
    size = min(42, max(7, 0.26 * n))
    fig, ax = plt.subplots(figsize=(size, size))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    fs = 4 if n > 90 else 6 if n > 40 else 8
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_per_class_f1(rows, path, name):
    plt = _mpl()
    r = sorted([x for x in rows if x["support"] > 0], key=lambda x: x["f1"])[:30]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(r) + 1.5))
    ax.barh([f"{x['name']} (n={x['support']})" for x in r], [x["f1"] for x in r], color="#b3392f")
    ax.set_xlim(0, 1); ax.set_xlabel("F1"); ax.set_title(f"30 weakest classes - {name}")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_roc(y, probs, C, path, name):
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    plt = _mpl()
    oh = np.zeros((len(y), C), dtype=np.uint8)
    oh[np.arange(len(y)), y] = 1
    fpr, tpr, _ = roc_curve(oh.ravel(), probs.ravel())
    pr, rc, _ = precision_recall_curve(oh.ravel(), probs.ravel())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(fpr, tpr, color="#2f7d4f"); ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
    ax[0].set_title(f"micro-average ROC (AUC {auc(fpr, tpr):.4f})"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[1].plot(rc, pr, color="#2f7d4f"); ax[1].set_title("micro-average precision-recall")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_reliability(data, path, T):
    plt = _mpl()
    fig, ax = plt.subplots(len(data), 2, figsize=(9, 4 * len(data)), squeeze=False)
    for r, (name, (p_raw, p_cal, y)) in enumerate(data.items()):
        for c, (p, tag) in enumerate(((p_raw, "before"), (p_cal, f"after temperature scaling (T={T:.2f})"))):
            ece, rows = ece_np(p, y)
            ax[r][c].plot([0, 1], [0, 1], "k--", lw=.8)
            ax[r][c].bar([(a + b) / 2 for a, b, *_ in rows], [x[2] for x in rows], width=1 / 15 * .9,
                         color="#2f7d4f", alpha=.85)
            ax[r][c].set_title(f"{name} - {tag}\nECE {ece:.4f}", fontsize=9)
            ax[r][c].set_xlabel("confidence"); ax[r][c].set_ylabel("accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_risk_coverage(data, thr, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (conf, ok) in data.items():
        cov, acc = risk_coverage(conf, ok)
        ax.plot(cov, acc, label=name)
    ax.set_xlabel("coverage (share of photos answered)"); ax.set_ylabel("accuracy of answered photos")
    ax.set_title(f"Selective prediction (app confidence cut-off {thr:.2f})"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_robustness(res, path):
    plt = _mpl()
    items = sorted(res.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(items) + 1.5))
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
            color=["#2f7d4f" if k == "clean" else "#8fa8c4" for k, _ in items][::-1])
    ax.set_xlim(0, 1); ax.set_xlabel("accuracy"); ax.set_title("Robustness to common photo problems")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _error_and_gradcam(model, d, y, logits, classes, tf, ck, device, T, out, ref):
    import numpy as np
    import torch
    from PIL import Image
    import textwrap
    plt = _mpl()
    import matplotlib
    probs = softmax_np(logits, T)
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:12]
    mean = torch.tensor(ck["mean"]).view(3, 1, 1)
    std = torch.tensor(ck["std"]).view(3, 1, 1)

    def show(i):
        with Image.open(d.fpath.iloc[i]) as im:
            x = tf(im.convert("RGB"))
        return x

    if len(wrong):
        n = len(wrong)
        fig, ax = plt.subplots(3, 4, figsize=(12, 10))
        for k, a_ in enumerate(ax.ravel()):
            a_.axis("off")
            if k < n:
                i = wrong[k]
                a_.imshow((show(i) * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
                a_.set_title(textwrap.fill(f"true: {pretty_label(classes[y[i]])}\npred: {pretty_label(classes[pred[i]])} "
                                           f"({conf[i]:.0%})", 34), fontsize=7)
        fig.suptitle(f"Most confident mistakes ({ref})"); fig.tight_layout()
        fig.savefig(out / "confident_mistakes.png", dpi=120); plt.close(fig)
    right = np.where(pred == y)[0]
    rng = np.random.default_rng(SEED)
    pick = list(rng.choice(right, size=min(6, len(right)), replace=False)) if len(right) else []
    pick += list(wrong[:max(0, 12 - len(pick))][:6])
    pick = pick[:12]
    if not pick:
        return
    xs = torch.stack([show(i) for i in pick]).to(device)
    cam, cls = gradcam_batch(model.eval(), xs)
    cmap = matplotlib.colormaps["jet"]
    fig, ax = plt.subplots(3, 4, figsize=(12, 10))
    for k, a_ in enumerate(ax.ravel()):
        a_.axis("off")
        if k < len(pick):
            img = (xs[k].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            ov = 0.55 * img + 0.45 * cmap(cam[k])[..., :3]
            a_.imshow(np.clip(ov, 0, 1))
            i = pick[k]
            a_.set_title(textwrap.fill(f"true: {pretty_label(classes[y[i]])}\npred: {pretty_label(classes[cls[k]])}", 34),
                         fontsize=7, color="#1b5e3a" if cls[k] == y[i] else "#b3392f")
    fig.suptitle(f"Grad-CAM: where the model looks ({ref}); green title = correct")
    fig.tight_layout(); fig.savefig(out / "gradcam.png", dpi=120); plt.close(fig)


def _write_report(M, path):
    def f(x, p=4):
        return "n/a" if x is None else f"{x:.{p}f}"
    L = [f"# Evaluation report - run `{M['run']}` ({M['arch']}, {M['n_classes']} classes, {M['img_size']}px)", ""]
    L += ["| set | images | accuracy | macro-F1 | top-3 | ROC-AUC | mAP | ECE | ECE after T-scaling |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in ("val", "test", "plantdoc_test"):
        if k in M:
            m = M[k]
            L.append(f"| {k} | {m['n']} | {f(m['accuracy'])} | {f(m['macro_f1'])} | {f(m['top3_accuracy'])} | "
                     f"{f(m['roc_auc_macro_ovr'])} | {f(m['mAP_macro'])} | {f(m['ece'])} | {f(m['ece_after_temperature'])} |")
    L += ["", "`plantdoc_test` = real-world photos from a different source that were never used for training or model selection.", ""]
    c = M.get("calibration", {})
    L += [f"Temperature T = {f(c.get('temperature'), 3)}; the app answers only when calibrated confidence >= "
          f"{f(c.get('conf_threshold'), 3)} (target accuracy {c.get('target_accuracy')}).", ""]
    for k in ("test", "plantdoc_test"):
        if k in M:
            s = M[k]["selective"]
            L.append(f"- {k}: answers {s['coverage']:.1%} of photos, accuracy on those {f(s['accuracy_when_answered'])}")
    for title, key in (("Accuracy by source (test)", "test_by_source"), ("Accuracy by domain (test)", "test_by_domain")):
        if key in M:
            L += ["", f"## {title}", "", "| group | n | accuracy |", "|---|---|---|"]
            L += [f"| {k} | {v['n']} | {f(v['accuracy'])} |" for k, v in M[key].items()]
    if "robustness_accuracy" in M:
        L += ["", "## Robustness (accuracy)", "", "| condition | accuracy |", "|---|---|"]
        L += [f"| {k} | {f(v)} |" for k, v in M["robustness_accuracy"].items()]
    if "speed" in M:
        L += ["", "## Size and speed", ""] + [f"- {k}: {v:.1f}" for k, v in M["speed"].items()]
    L += ["", "## Figures", "",
          "confusion_test.png, confusion_plantdoc_test.png, worst_classes_*.png, roc_pr_*.png, reliability.png, "
          "risk_coverage.png, confident_mistakes.png, gradcam.png, robustness.png, per_class_*.csv, top_confusions_*.csv"]
    Path(path).write_text("\n".join(L), encoding="utf-8")


# 11. Farmer advice
_LOCAL = "Use fungicide or bactericide only if a product is approved for this crop where you farm; follow the label and ask your local agricultural extension officer."
ADVICE = [
    (r"^healthy$", "No disease signs found", "This leaf looks healthy in the photo.",
     ["Keep checking plants every week, and after heavy rain.", "Water at the base of the plant; avoid wetting leaves.",
      "Remove fallen leaves and weeds around the plants.", "Check other leaves too: one healthy leaf does not clear the whole plant."]),
    (r"late blight", "Late blight", "A fast-spreading water mould disease of potato and tomato in cool, wet weather.",
     ["Act at once: remove and destroy affected leaves or plants (do not compost them).", "Avoid overhead watering; water in the morning.",
      "Space plants for airflow and stake tomatoes.", "Do not plant potato or tomato in the same field next season.", _LOCAL]),
    (r"early blight", "Early blight", "A fungal disease that makes brown rings on older leaves, mostly in warm, humid weather.",
     ["Remove the lowest infected leaves.", "Mulch the soil so spores do not splash onto leaves.", "Keep plants well fed and watered; stressed plants get it worse.",
      "Rotate crops for 2 to 3 years.", _LOCAL]),
    (r"powdery mildew", "Powdery mildew", "A fungus that forms a white powder on leaves.",
     ["Remove badly affected leaves.", "Improve airflow: wider spacing, light pruning.", "Avoid too much nitrogen fertiliser.", "Choose resistant varieties next season.", _LOCAL]),
    (r"downy mildew", "Downy mildew", "Yellow patches on top of the leaf with grey or purple fuzz underneath, in humid weather.",
     ["Remove affected leaves and destroy them.", "Water early in the day and keep leaves dry.", "Improve airflow.", "Rotate crops and use resistant varieties.", _LOCAL]),
    (r"rust", "Rust", "A fungus that forms orange, brown or black pustules on leaves.",
     ["Remove and destroy affected leaves.", "Avoid wetting leaves; improve airflow.", "Clear fallen leaves at the end of the season.", "Use resistant varieties.", _LOCAL]),
    (r"mosaic|yellow leaf curl|virus|streak|mottle|leafroll|leaf roll", "Virus disease",
     "Viruses cannot be cured. They spread by insects (whitefly, aphids), tools and hands.",
     ["Remove and destroy infected plants early.", "Control insects that spread the virus (whitefly, aphids).", "Plant virus-free seed, cuttings or seedlings; use resistant varieties.",
      "Wash hands and tools after touching infected plants.", "Remove weeds that hold the virus."]),
    (r"bacterial|canker|halo blight|fire blight", "Bacterial disease",
     "Bacteria spread by splashing water, tools and hands, and enter through wounds.",
     ["Do not work in the field when plants are wet.", "Remove infected leaves or plants; clean tools between plants.", "Use clean seed or seedlings; rotate crops.",
      "Wilt diseases have no cure: remove the plant.", "Copper products help only in some crops; ask your extension officer."]),
    (r"scab", "Scab", "A fungal disease that makes dark, rough spots on leaves and fruit.",
     ["Rake up and destroy fallen leaves.", "Prune for airflow.", "Use resistant varieties.", _LOCAL]),
    (r"black rot|esca|black measles", "Black rot / esca", "Fungal diseases that cause dark spots, rotting fruit and dying shoots.",
     ["Remove and destroy infected leaves, fruit and shoots.", "Prune for airflow and sunlight.", "Clear mummified fruit.", _LOCAL]),
    (r"leaf mold|mould", "Leaf mold", "A fungus that grows in high humidity, mainly in greenhouses and dense crops.",
     ["Lower humidity: ventilate and space plants.", "Remove affected leaves.", "Water at the base.", "Use resistant varieties.", _LOCAL]),
    (r"mite", "Spider mites", "Tiny pests that suck sap, causing speckled, yellow leaves; worse in hot, dry weather.",
     ["Spray leaves with a strong jet of water, underside too.", "Keep plants watered; dusty, dry plants get more mites.", "Protect natural enemies; avoid broad-spectrum insecticides.",
      "Use a miticide only if your extension officer recommends one."]),
    (r"greening", "Citrus greening (HLB)", "A serious bacterial disease spread by the citrus psyllid. It has no cure.",
     ["Report suspected trees to your agriculture office.", "Remove infected trees as advised locally.", "Control psyllids.", "Plant only certified disease-free trees."]),
    (r"blast|sheath blight", "Rice blast / sheath blight", "Fungal rice diseases favoured by heavy nitrogen and humid weather.",
     ["Avoid too much nitrogen fertiliser.", "Use resistant varieties and certified seed.", "Manage water and plant spacing.", _LOCAL]),
    (r"smut|charcoal", "Smut / charcoal rot", "Fungal diseases that form galls or black growth on the plant.",
     ["Remove galls before they burst; destroy them.", "Rotate crops.", "Use certified, treated seed.", _LOCAL]),
    (r"root rot|wilt|panama|pocket", "Rot / wilt disease", "Soil-borne diseases that damage roots or fruit and wilt the plant.",
     ["Improve drainage; avoid over-watering.", "Remove and destroy infected plants.", "Rotate crops; avoid replanting in the same spot.", "Use clean tools and resistant varieties."]),
    (r"septoria|spot|cercospora|anthracnose|target|frog eye|tar spot|cavity|chlorotic|alternaria|blight|scorch|curl",
     "Fungal leaf disease", "A group of fungal diseases that cause spots, blotches or curling on leaves.",
     ["Remove and destroy affected leaves.", "Avoid overhead watering; water early in the day.", "Space plants for airflow.",
      "Clear crop residue after harvest and rotate crops.", _LOCAL]),
]
_ADVICE_DEFAULT = ("Unclear problem", "The photo shows signs of stress or disease that need a closer look.",
                   ["Remove severely affected leaves.", "Avoid wetting leaves; improve airflow.", "Take clear photos (leaf top and underside) and show them to your local agricultural extension officer.",
                    "Check for insects, nutrient shortage and water stress; these look similar to disease."])


def advice_for(cond: str):
    c = (cond or "").lower()
    for pat, title, about, acts in ADVICE:
        if re.search(pat, c):
            return title, about, acts
    return _ADVICE_DEFAULT


# 12. Export
def _prep(img, size, mean, std):
    import numpy as np
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = int(round(size * 1.14))
    sc = s / min(w, h)
    img = img.resize((max(s, round(w * sc)), max(s, round(h * sc))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    img = img.crop((l, t, l + size, t + size))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return a.transpose(2, 0, 1)[None].astype(np.float32), img


def _ort_latency(path, size, n=20):
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    for _ in range(3):
        s.run(None, {"input": x})
    t = time.time()
    for _ in range(n):
        s.run(None, {"input": x})
    return (time.time() - t) / n * 1000


def export_cmd(a):
    import numpy as np
    import torch
    import onnxruntime as ort
    from PIL import Image
    run = a.run or _latest_run()
    ck, model, run_dir = load_run(run)
    out = Path(a.out) if a.out else DEPLOY_DIR
    out.mkdir(parents=True, exist_ok=True)
    size, classes = ck["img_size"], ck["classes"]
    mean, std = ck["mean"], ck["std"]
    model = model.cpu().float().eval()
    onnx_path = out / "model.onnx"
    log(f"== exporting '{run}' to ONNX (opset 17)")
    with torch.no_grad():
        torch.onnx.export(model, torch.randn(1, 3, size, size), str(onnx_path), input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, do_constant_folding=True)
    df = load_manifest()
    cset = set(classes)
    chk = df[(df.split == "test") & df.label.isin(cset)]
    if ck.get("sources"):
        chk = chk[chk.src.isin(ck["sources"])]
    chk = chk.sample(n=min(48, len(chk)), random_state=SEED) if len(chk) else chk
    xs = [_prep(Image.open(p), size, mean, std)[0] for p in chk.fpath]
    info = {"onnx_fp32_mb": onnx_path.stat().st_size / 2**20}
    if xs:
        X = np.concatenate(xs)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        lo_ort = sess.run(None, {"input": X})[0]
        with torch.no_grad():
            lo_pt = model(torch.from_numpy(X)).numpy()
        info["onnx_max_abs_logit_diff_vs_pytorch"] = float(np.abs(lo_ort - lo_pt).max())
        info["onnx_prediction_agreement"] = float((lo_ort.argmax(1) == lo_pt.argmax(1)).mean())
        log(f"   ONNX vs PyTorch: max |logit diff| {info['onnx_max_abs_logit_diff_vs_pytorch']:.2e}   "
            f"same prediction {info['onnx_prediction_agreement']:.1%}")
    info["onnx_fp32_cpu_latency_ms"] = _ort_latency(onnx_path, size)
    int8 = out / "model_int8.onnx"
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(onnx_path), str(int8), weight_type=QuantType.QInt8)
        info["onnx_int8_mb"] = int8.stat().st_size / 2**20
        if xs:
            s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
            l8 = s8.run(None, {"input": X})[0]
            info["int8_prediction_agreement_with_fp32"] = float((l8.argmax(1) == lo_ort.argmax(1)).mean())
        info["onnx_int8_cpu_latency_ms"] = _ort_latency(int8, size)
        log(f"   int8: {info['onnx_int8_mb']:.0f} MB, agreement with fp32 {info.get('int8_prediction_agreement_with_fp32', float('nan')):.1%}, "
            f"{info['onnx_int8_cpu_latency_ms']:.0f} ms/img")
        if info.get("int8_prediction_agreement_with_fp32", 1.0) < 0.95:
            log("   [!] int8 changes too many predictions - the app will use fp32 by default")
    except Exception as e:
        log(f"   [skip] int8 quantization: {e}")
        if int8.is_file():
            int8.unlink()
    log(f"   fp32: {info['onnx_fp32_mb']:.0f} MB, {info['onnx_fp32_cpu_latency_ms']:.0f} ms/img (CPU, 4 threads)")
    if info["onnx_fp32_mb"] > 95:
        log("   [!] model.onnx is over GitHub's 100 MB file limit -> use Git LFS, or commit model_int8.onnx")

    ev = read_json(run_dir / "eval" / "metrics.json", {}) or {}
    cal = ev.get("calibration") or read_json(run_dir / "eval" / "calibration.json", {}) or {}
    if not cal:
        log("   [!] no evaluation found - run evaluate first for calibrated confidence. Using defaults.")
    cfg = {"run": run, "arch": ck["arch"], "img_size": size, "mean": mean, "std": std, "classes": classes,
           "temperature": cal.get("temperature", 1.0), "conf_threshold": cal.get("conf_threshold", 0.5),
           "maxlogit_threshold": cal.get("maxlogit_threshold", float("-inf")),
           "metrics": {k: {"accuracy": ev[k]["accuracy"], "macro_f1": ev[k]["macro_f1"], "n": ev[k]["n"]}
                       for k in ("test", "plantdoc_test") if k in ev},
           "export": info}
    write_json(out / "model_config.json", cfg)
    sd = out / "samples"
    sd.mkdir(exist_ok=True)
    for f in sd.glob("*.jpg"):
        f.unlink()
    pool = df[(df.split == "ext_test") & df.label.isin(cset)]
    if len(pool) == 0:
        pool = df[(df.split == "test") & df.label.isin(cset)]
    seen = set()
    for _, r in pool.sample(frac=1.0, random_state=SEED).iterrows():
        if r.label in seen or len(seen) >= 10:
            continue
        seen.add(r.label)
        shutil.copy(r.fpath, sd / f"{r.label}.jpg")
    (out / "requirements.txt").write_text("streamlit>=1.35\nonnxruntime>=1.18\nnumpy>=1.26\npillow>=10.0\nmatplotlib>=3.8\n")
    (out / "MODEL_CARD.md").write_text(_model_card(cfg, ev, run_dir), encoding="utf-8")
    log(f"== deploy folder ready: {out}")
    log("   test locally:   streamlit run leafguard.py")
    log("   deploy: push leafguard.py + deploy/ + requirements.txt (copy it to the repo root) to GitHub, then Streamlit Community Cloud")
    return 0


def _model_card(cfg, ev, run_dir):
    rep = read_json(WORK_DIR / "data_report.json", {}) or {}
    info = read_json(run_dir / "run_info.json", {}) or {}
    t, e = ev.get("test", {}), ev.get("plantdoc_test", {})
    g = lambda d, k: f"{d[k]:.4f}" if d and d.get(k) is not None else "n/a"
    L = ["# Model card - LeafGuard plant disease classifier", "",
         "## Model", f"- Architecture: `{cfg['arch']}` (ImageNet pre-trained, fine-tuned), input {cfg['img_size']}x{cfg['img_size']} px",
         f"- Classes: {len(cfg['classes'])} (crop + condition, e.g. `tomato__late_blight`)",
         "- Output: calibrated probabilities (temperature scaling); the app refuses to answer when confidence is low",
         "", "## Intended use",
         "Decision support for farmers, students and extension workers: photograph one leaf, get a likely diagnosis and general next steps.",
         "Not a replacement for an agronomist or a lab test. Not for regulatory or commercial crop-protection decisions.", "",
         "## Training data",
         "Six public datasets, labels unified into one class list (see `label_map.csv`): PlantVillage (lab photos), PlantWild, PlantDoc, FieldPlant "
         "(leaf boxes cropped), Cassava Leaf Disease (Kaggle), Plant Pathology 2021 (single-label photos only).",
         f"- Images used: {rep.get('n_images', 'n/a')} after removing exact duplicates; splits made per source photo group so near-copies cannot leak.",
         "- PlantDoc's official test set is held out completely: never used for training, tuning or model selection.", "",
         "## Evaluation", "| set | n | accuracy | macro-F1 |", "|---|---|---|---|",
         f"| internal test | {t.get('n', 'n/a')} | {g(t, 'accuracy')} | {g(t, 'macro_f1')} |",
         f"| PlantDoc test (real-world, held out) | {e.get('n', 'n/a')} | {g(e, 'accuracy')} | {g(e, 'macro_f1')} |", "",
         "Full report with confusion matrices, calibration, robustness, Grad-CAM: `eval/report.md`.", "",
         "## Limitations",
         "- Trained on public datasets that over-represent a few crops and lab-style photos; accuracy on other crops, varieties, cameras and countries can be lower.",
         "- One leaf per photo works best. Several leaves, whole fields, fruits and stems are outside the training data.",
         "- The classifier has no 'not a plant' class. The confidence check rejects many unrelated photos but not all.",
         "- Multiple diseases on one leaf, nutrient shortage and pest damage can be mistaken for a single disease.",
         "- Treatment text is general guidance only; confirm with a local agricultural extension officer.", "",
         "## Data licences (read before sharing or using commercially)",
         "Several source datasets are licensed non-commercial and/or share-alike (for example PlantVillage CC BY-NC-SA 4.0). "
         "Treat this model and demo as research / education use unless you have checked every dataset licence. Do not re-publish the raw datasets.", "",
         "## Training setup", f"```\n{json.dumps(info.get('args', {}), indent=2)}\n```"]
    return "\n".join(L)


# 13. Compare, domain gap, all
def compare_cmd(a):
    rows = []
    for mf in sorted(RUNS_DIR.glob("*/eval/metrics.json")):
        m = read_json(mf, {})
        t, e, s = m.get("test", {}), m.get("plantdoc_test", {}), m.get("speed", {})
        rows.append({"run": m.get("run", mf.parents[1].name), "arch": m.get("arch"), "img": m.get("img_size"),
                     "classes": m.get("n_classes"), "params_M": s.get("params_million"),
                     "test_acc": t.get("accuracy"), "test_macroF1": t.get("macro_f1"),
                     "plantdoc_acc": e.get("accuracy"), "plantdoc_macroF1": e.get("macro_f1"),
                     "test_ECE": t.get("ece"), "gpu_img_per_s": s.get("gpu_images_per_s_bs64"),
                     "cpu_ms": s.get("cpu_latency_ms_bs1_4threads")})
    if not rows:
        log("no evaluated runs yet (train + evaluate first)")
        return 1
    fmt = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) and abs(v) < 10 else
                                           f"{v:.1f}" if isinstance(v, float) else str(v))
    cols = list(rows[0].keys())
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    L += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    txt = "\n".join(L)
    (WORK_DIR / "comparison.md").write_text(txt, encoding="utf-8")
    with open(WORK_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log(txt)
    log(f"\nsaved: {WORK_DIR / 'comparison.md'}")
    return 0


def domain_gap_cmd(a):
    b = copy.copy(a)
    b.sources, b.run_name = "plantvillage", a.run_name or f"lab_only_{a.arch}"
    b.run = b.run_name
    train_cmd(b)
    b.no_robustness, b.tta = True, False
    evaluate_cmd(b)
    return compare_cmd(b)


def all_cmd(a):
    if not MANIFEST.is_file() or a.rebuild:
        if prepare_cmd(a):
            return 1
    if train_cmd(a):
        return 1
    a.run = a.run_name or a.arch
    if evaluate_cmd(a):
        return 1
    return export_cmd(a)


# 14. Streamlit app
def _occlusion_heatmap(sess, x, cls, T, patch_frac=0.25, stride_frac=0.125):
    import numpy as np
    size = x.shape[-1]
    p = max(8, int(size * patch_frac))
    step = max(4, int(size * stride_frac))
    pos = list(range(0, size - p + 1, step))
    batch, coords = [], []
    for yy in pos:
        for xx in pos:
            xb = x.copy()
            xb[:, :, yy:yy + p, xx:xx + p] = 0.0
            batch.append(xb[0])
            coords.append((yy, xx))
    pr = softmax_np(sess.run(None, {"input": np.stack(batch)})[0], T)[:, cls]
    base = softmax_np(sess.run(None, {"input": x})[0], T)[0, cls]
    heat, cnt = np.zeros((size, size)), np.zeros((size, size))
    for (yy, xx), pv in zip(coords, pr):
        heat[yy:yy + p, xx:xx + p] += base - pv
        cnt[yy:yy + p, xx:xx + p] += 1
    heat = np.clip(heat / np.maximum(cnt, 1), 0, None)
    return heat / (heat.max() + 1e-8)


def _fetch_model(url, dest):
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def run_app():
    import numpy as np
    import streamlit as st
    from PIL import Image
    import onnxruntime as ort

    st.set_page_config(page_title="LeafGuard", page_icon="🌿", layout="wide")
    st.markdown("""
    <style>
      .block-container {max-width: 1100px; padding-top: 2rem;}
      .lg-result {border-left: 8px solid var(--c); padding: 14px 20px; border-radius: 4px; background: rgba(128,128,128,.10);}
      .lg-crop {font-size: 1rem; opacity: .75; margin: 0;}
      .lg-name {font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 2px 0 6px 0;}
      .lg-conf {font-size: 1rem; margin: 0;}
      .lg-note {font-size: .85rem; opacity: .7;}
    </style>""", unsafe_allow_html=True)

    cfg_path = DEPLOY_DIR / "model_config.json"
    if not cfg_path.is_file():
        st.error(f"Model files not found in `{DEPLOY_DIR}`. Run `python leafguard.py export` first, "
                 "and keep the `deploy` folder next to this file.")
        st.stop()
    cfg = read_json(cfg_path)
    classes, size = cfg["classes"], cfg["img_size"]
    fp32, int8 = DEPLOY_DIR / "model.onnx", DEPLOY_DIR / "model_int8.onnx"
    url = os.environ.get("LEAFGUARD_MODEL_URL") or cfg.get("model_url")
    if not fp32.is_file() and url:
        with st.spinner("Downloading the model (first start only)..."):
            try:
                _fetch_model(url, fp32)
            except Exception:
                pass
    if not fp32.is_file() and not int8.is_file():
        st.error("No model file found. Add deploy/model_int8.onnx, or set model_url in deploy/model_config.json.")
        st.stop()

    st.title("LeafGuard")
    st.write("Photograph one leaf. Get the most likely disease and what to do next.")

    with st.sidebar:
        st.header("Good photo = good answer")
        st.markdown("- One leaf, filling most of the frame\n- In focus, in daylight, no shadow or flash glare\n"
                    "- Show the sick part; the plain side facing up\n- Take 2 or 3 photos of different leaves")
        st.header("Settings")
        topk = st.slider("Possible matches to show", 1, 5, 3)
        explain = st.checkbox("Show where the model looked (slower)", value=False)
        use_int8 = (st.checkbox("Compact model (smaller, slightly less exact)", value=False)
                    if fp32.is_file() and int8.is_file() else int8.is_file())
        st.header("About")
        mt = cfg.get("metrics", {})
        st.caption(f"{len(classes)} crop/condition classes. Model: {cfg['arch']}.")
        if "test" in mt:
            st.caption(f"Accuracy on held-out test photos: {mt['test']['accuracy']:.1%}")
        if "plantdoc_test" in mt:
            st.caption(f"Accuracy on real-world PlantDoc photos: {mt['plantdoc_test']['accuracy']:.1%}")
        st.caption("Research and education tool. Not a substitute for an agronomist.")

    @st.cache_resource(show_spinner=False)
    def load_session(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = min(4, os.cpu_count() or 1)
        return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

    sess = load_session(str(int8 if use_int8 else fp32))

    img, src_name = None, ""
    t_up, t_cam, t_demo = st.tabs(["Upload a photo", "Use the camera", "Try a sample"])
    with t_up:
        f = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if f is not None:
            img, src_name = Image.open(f), f.name
    with t_cam:
        c = st.camera_input("Take a photo of one leaf")
        if c is not None and img is None:
            img, src_name = Image.open(c), "camera"
    with t_demo:
        samples = sorted((DEPLOY_DIR / "samples").glob("*.jpg"))
        if samples:
            names = {pretty_label(p.stem): p for p in samples}
            choice = st.selectbox("Sample leaf", ["-"] + list(names))
            if choice != "-" and img is None:
                img, src_name = Image.open(names[choice]), choice
        else:
            st.caption("No sample photos in this deployment.")

    if img is None:
        st.info("Add a photo above to start.")
        return

    x, shown = _prep(img, size, cfg["mean"], cfg["std"])
    logits = sess.run(None, {"input": x})[0]
    T = cfg.get("temperature", 1.0)
    p = softmax_np(logits, T)[0]
    order = np.argsort(-p)
    top = int(order[0])
    conf = float(p[top])
    ml_thr = cfg.get("maxlogit_threshold", float("-inf"))
    unsure = conf < cfg.get("conf_threshold", 0.5) or float(logits[0].max()) < (ml_thr if ml_thr is not None else -1e9)
    crop, cond = split_key(classes[top])
    healthy = cond == "healthy"

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.image(shown.resize((448, 448)), caption="What the model sees (centre of your photo)")
    with right:
        if unsure:
            colour, head, sub = "#B7791F", "Not sure", f"Best guess: {crop.capitalize()}, {cond} ({conf:.0%})"
        elif healthy:
            colour, head, sub = "#2E7D4F", "Looks healthy", f"{crop.capitalize()} ({conf:.0%} confident)"
        else:
            colour, head, sub = "#B3392F", cond.capitalize(), f"{crop.capitalize()} ({conf:.0%} confident)"
        st.markdown(f"<div class='lg-result' style='--c:{colour}'><p class='lg-crop'>{sub}</p>"
                    f"<p class='lg-name'>{head}</p></div>", unsafe_allow_html=True)
        if unsure:
            st.warning("The model is not confident enough to name a disease. Take a new photo: one leaf, in focus, "
                       "in daylight, filling most of the picture. Try a different leaf too.")
        st.markdown("**Possible matches**")
        for i in order[:topk]:
            c_, d_ = split_key(classes[int(i)])
            st.progress(float(p[i]), text=f"{c_.capitalize()}: {d_}  ({p[i]:.0%})")

    if not unsure:
        title, about, acts = advice_for(cond)
        st.subheader("What to do next" if not healthy else "Keep it healthy")
        st.write(about)
        st.markdown("\n".join(f"- {a_}" for a_ in acts))
        if not healthy:
            st.caption("General guidance only. Disease control rules and approved products differ by country: "
                       "confirm with your local agricultural extension officer.")

    if explain:
        with st.spinner("Checking which parts of the leaf decided the answer..."):
            import matplotlib
            heat = _occlusion_heatmap(sess, x, top, T)
            ov = 0.55 * np.asarray(shown, dtype=np.float32) / 255 + 0.45 * matplotlib.colormaps["jet"](heat)[..., :3]
        ov_img = Image.fromarray((np.clip(ov, 0, 1) * 255).astype(np.uint8)).resize((448, 448))
        st.image(ov_img, caption="Warm colours = parts of the photo that mattered most for this answer")
    st.markdown("<p class='lg-note'>Photos are processed in memory and not stored.</p>", unsafe_allow_html=True)


# 15. Command line
def build_parser():
    ap = argparse.ArgumentParser(prog="leafguard.py", description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def add_train_args(p):
        p.add_argument("--arch", default="convnext_tiny", help="any timm model: convnext_tiny, efficientnet_b2, resnet50, mobilenetv3_large_100 ...")
        p.add_argument("--run-name", default=None, help="folder name for this run (default = arch)")
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--img-size", type=int, default=224)
        p.add_argument("--lr", type=float, default=2e-4, help="backbone learning rate (head gets x10)")
        p.add_argument("--head-mult", type=float, default=10.0)
        p.add_argument("--wd", type=float, default=0.05)
        p.add_argument("--warmup-epochs", type=float, default=1.0)
        p.add_argument("--label-smoothing", type=float, default=0.1)
        p.add_argument("--drop", type=float, default=0.2)
        p.add_argument("--drop-path", type=float, default=0.1)
        p.add_argument("--ema-decay", type=float, default=0.999)
        p.add_argument("--samples-per-epoch", type=int, default=60000, help="0 = whole train set")
        p.add_argument("--field-boost", type=float, default=1.5, help="sampling weight of field photos vs lab photos")
        p.add_argument("--sources", default="all", help="all, or comma list e.g. plantvillage,plantwild")
        p.add_argument("--workers", type=int, default=0, help="0 = auto")
        p.add_argument("--patience", type=int, default=6)
        p.add_argument("--no-pretrained", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--seed", type=int, default=SEED)

    def add_eval_args(p):
        p.add_argument("--run", default=None, help="run folder name (default: newest)")
        p.add_argument("--tta", action="store_true", help="test-time augmentation (horizontal flip)")
        p.add_argument("--no-robustness", action="store_true")
        p.add_argument("--robust-n", type=int, default=1500)
        p.add_argument("--target-acc", type=float, default=0.95, help="accuracy the app should reach on photos it answers")

    sub.add_parser("check", help="verify environment and dataset folders")
    p = sub.add_parser("prepare", help="scan, unify labels, cache, dedupe, split")
    p.add_argument("--min-images", type=int, default=MIN_IMAGES_PER_CLASS)
    p = sub.add_parser("train"); add_train_args(p)
    p = sub.add_parser("evaluate"); add_eval_args(p)
    p = sub.add_parser("export"); p.add_argument("--run", default=None); p.add_argument("--out", default=None)
    sub.add_parser("compare", help="table of all evaluated runs")
    p = sub.add_parser("domain-gap", help="lab-only baseline: PlantVillage -> PlantDoc"); add_train_args(p); add_eval_args(p)
    p.set_defaults(epochs=8)
    p = sub.add_parser("all", help="prepare -> train -> evaluate -> export"); add_train_args(p); add_eval_args(p)
    p.add_argument("--rebuild", action="store_true", help="force prepare even if manifest exists")
    p.add_argument("--min-images", type=int, default=MIN_IMAGES_PER_CLASS)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 0
    if a.cmd == "all":
        a.run = a.run_name or a.arch
    fn = {"check": check_cmd, "prepare": prepare_cmd, "train": train_cmd, "evaluate": evaluate_cmd,
          "export": export_cmd, "compare": compare_cmd, "domain-gap": domain_gap_cmd, "all": all_cmd}[a.cmd]
    return fn(a) or 0


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    run_app()
elif __name__ == "__main__":
    sys.exit(main())
