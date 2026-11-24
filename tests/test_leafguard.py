import csv
import importlib.util
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_counter = itertools.count()


def load_module(env=None):
    for k, v in (env or {}).items():
        os.environ[k] = str(v)
    name = f"leafguard_under_test_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "leafguard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def key(lg, raw, default_crop=None):
    return lg.canon_label(raw, default_crop)[2]


def test_label_unification_across_datasets():
    lg = load_module()
    same = [
        ["Tomato___Late_blight", "tomato late blight", "Tomato leaf late blight"],
        ["Corn_(maize)___Common_rust_", "corn rust leaf", "corn rust"],
        ["Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot", "bell pepper leaf spot"],
        ["Apple___Cedar_apple_rust", "Apple rust leaf", "apple rust"],
        ["Orange___Haunglongbing_(Citrus_greening)", "citrus greening disease"],
        ["Tomato___healthy", "tomato healthy", "Tomato leaf"],
        ["Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight", "corn northern leaf blight"],
        ["Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus", "tomato yellow leaf curl virus"],
    ]
    for group in same:
        assert len({key(lg, r) for r in group}) == 1, group
    assert key(lg, "Tomato___Late_blight") == "tomato__late_blight"
    assert key(lg, "Cassava Mosaic Disease (CMD)") == key(lg, "Cassava Mosaic") == "cassava__mosaic_disease"
    assert key(lg, "Healthy", "cassava") == "cassava__healthy"
    assert key(lg, "frog_eye_leaf_spot", "apple") == "apple__frog_eye_leaf_spot"


def test_label_unification_keeps_different_diseases_apart():
    lg = load_module()
    assert key(lg, "Tomato___Early_blight") != key(lg, "Tomato___Late_blight")
    assert key(lg, "Potato___Late_blight") != key(lg, "Tomato___Late_blight")
    assert key(lg, "Cassava Brown Leaf Spot") != key(lg, "Cassava Brown Streak Disease (CBSD)")
    assert lg.split_key("tomato__late_blight") == ("tomato", "late blight")
    assert lg.pretty_label("apple__healthy") == "Apple: healthy"


def test_metrics_perfect_and_calibration():
    lg = load_module()
    y = np.arange(6) % 3
    logits = np.eye(3)[y] * 10
    m = lg.compute_metrics(y, logits)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["top3_accuracy"] == 1.0
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 3000)
    lg_over = rng.normal(0, 1, (3000, 10))
    lg_over[np.arange(3000), y] += 2.0
    lg_over *= 4.0
    T = lg.fit_temperature(lg_over, y)
    assert T > 1.5
    assert lg.nll_np(lg_over, y, T) < lg.nll_np(lg_over, y, 1.0)
    p = lg.softmax_np(lg_over, T)
    assert np.allclose(p.sum(1), 1.0)
    assert lg.ece_np(p, y)[0] < lg.ece_np(lg.softmax_np(lg_over, 1.0), y)[0]


def test_confidence_threshold_reaches_target_accuracy():
    lg = load_module()
    rng = np.random.default_rng(1)
    conf = rng.random(5000)
    correct = rng.random(5000) < (0.5 + 0.5 * conf)
    thr, coverage = lg.pick_conf_threshold(conf, correct, 0.9)
    kept = conf >= thr
    assert 0 < coverage <= 1
    assert correct[kept].mean() >= 0.9 - 0.02


def test_advice_and_preprocessing():
    lg = load_module()
    assert lg.advice_for("late blight")[0] == "Late blight"
    assert lg.advice_for("healthy")[0] == "No disease signs found"
    assert lg.advice_for("something unknown")[0] == "Unclear problem"
    img = Image.fromarray((np.random.default_rng(0).random((300, 420, 3)) * 255).astype(np.uint8))
    x, shown = lg._prep(img, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and shown.size == (224, 224)


def test_command_line_parsing():
    lg = load_module()
    ap = lg.build_parser()
    assert ap.parse_args(["train"]).epochs == 20
    assert ap.parse_args(["domain-gap"]).epochs == 8
    assert ap.parse_args(["train", "--arch", "efficientnet_b0", "--resume"]).resume is True
    for cmd in (["check"], ["prepare"], ["evaluate", "--tta"], ["export"], ["compare"], ["all", "--rebuild"]):
        assert ap.parse_args(cmd).cmd == cmd[0]


def make_image(path, seed, size=(120, 90)):
    a = (np.random.default_rng(seed).random((size[1], size[0], 3)) * 255).astype(np.uint8)
    a[: size[1] // 2] //= 2
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, quality=90)


def build_synthetic_datasets(root):
    n = itertools.count(1)
    pv = root / "archive" / "plantvillage dataset" / "color"
    for cls in ["Tomato___Late_blight", "Tomato___healthy", "Apple___Apple_scab", "Apple___healthy"]:
        for i in range(30):
            make_image(pv / cls / f"{i}.jpg", next(n))
    for i in range(30):
        (pv / f"Dummy{i}___healthy").mkdir(parents=True, exist_ok=True)
    for split, k in (("train", 14), ("val", 4), ("test", 4)):
        for cls in ["tomato late blight", "tomato leaf", "apple scab", "apple leaf"]:
            for i in range(k):
                make_image(root / "archive (1)" / split / cls / f"{i}.jpg", next(n))
    fp = root / "archive (2)"
    (fp / "train" / "images").mkdir(parents=True)
    (fp / "train" / "labels").mkdir(parents=True)
    (fp / "data.yaml").write_text("nc: 3\nnames: ['Tomato healthy','Tomato blight leaf','Cassava Healthy']\n")
    for j in range(12):
        for aug in range(3):
            name = f"photo{j}_jpg.rf.{aug}abc{j}"
            make_image(fp / "train" / "images" / f"{name}.jpg", next(n), (200, 200))
            (fp / "train" / "labels" / f"{name}.txt").write_text(f"{j % 3} 0.5 0.5 0.6 0.6\n{(j + 1) % 3} 0.2 0.2 0.3 0.3\n")
    cs = root / "cassava-leaf-disease-classification"
    (cs / "train_images").mkdir(parents=True)
    (cs / "label_num_to_disease_map.json").write_text(json.dumps(
        {"0": "Cassava Bacterial Blight (CBB)", "1": "Cassava Brown Streak Disease (CBSD)", "4": "Healthy"}))
    with open(cs / "train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label"])
        for i in range(30):
            make_image(cs / "train_images" / f"{i}.jpg", next(n))
            w.writerow([f"{i}.jpg", [0, 1, 4][i % 3]])
    pd_ = root / "PlantDoc-Dataset-master"
    for split in ("train", "test"):
        for cls in ["Tomato leaf late blight", "Tomato leaf", "Apple Scab Leaf"]:
            for i in range(10 if split == "train" else 4):
                make_image(pd_ / split / cls / f"{i}.jpg", next(n))
    Image.open(pd_ / "test" / "Tomato leaf" / "0.jpg").save(pv / "Tomato___healthy" / "exact_copy.jpg", quality=95)
    Image.open(pd_ / "test" / "Apple Scab Leaf" / "1.jpg").resize((110, 82)).save(
        pv / "Apple___Apple_scab" / "near_copy.jpg", quality=60)
    pp = root / "plant-pathology-2021-fgvc8"
    (pp / "train_images").mkdir(parents=True)
    with open(pp / "train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "labels"])
        for i in range(40):
            make_image(pp / "train_images" / f"{i}.jpg", next(n), (500, 380))
            w.writerow([f"{i}.jpg", ["healthy", "scab", "scab frog_eye_leaf_spot", "complex", "rust"][i % 5]])


def test_prepare_pipeline_is_leak_free(tmp_path):
    build_synthetic_datasets(tmp_path)
    lg = load_module({"LEAFGUARD_ROOT": tmp_path, "LEAFGUARD_WORK": tmp_path / "out"})
    assert lg.prepare_cmd(SimpleNamespace(min_images=3)) == 0
    rows = list(csv.DictReader(open(tmp_path / "out" / "manifest.csv", newline="")))
    report = json.load(open(tmp_path / "out" / "data_report.json"))
    removed = report["exact_duplicates_removed"] + report["near_duplicates_of_plantdoc_test_removed"]
    assert removed >= 2
    assert sum(r["split"] == "ext_test" for r in rows) == 12
    assert not any(r["src"] == "plantvillage" and r["split"] == "ext_test" for r in rows)
    spans = defaultdict(set)
    for r in rows:
        if r["group"]:
            spans[r["group"]].add(r["split"])
    assert spans and all(len(v) == 1 for v in spans.values())
    assert {"train", "val", "test"} <= {r["split"] for r in rows}
    assert (tmp_path / "out" / "label_map.csv").is_file()
    classes = json.load(open(tmp_path / "out" / "classes.json"))["classes"]
    assert "tomato__late_blight" in classes and "cassava__healthy" in classes


def test_model_download_helper(tmp_path):
    lg = load_module()
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1000)
    dest = tmp_path / "nested" / "model.onnx"
    lg._fetch_model(src.as_uri(), dest)
    assert dest.stat().st_size == 1000 and not dest.with_name("model.onnx.part").exists()


def test_deployment_bundle_is_complete_and_matches_the_main_checkpoint():
    import json

    deploy = ROOT / "deploy"
    cfg = json.loads((deploy / "model_config.json").read_text(encoding="utf-8"))
    assert cfg["run"] == "convnext_tiny"
    assert len(cfg["classes"]) == 109
    assert len(set(cfg["classes"])) == 109
    assert cfg["img_size"] == 224
    assert abs(cfg["export"]["int8_prediction_agreement_with_fp32"] - 0.9791666667) < 1e-6

    model = deploy / "model_int8.onnx"
    assert model.is_file() and model.stat().st_size > 20_000_000
    assert not (deploy / "model.onnx").exists()

    samples = sorted((deploy / "samples").glob("*.jpg"))
    assert len(samples) == 10
    for sample in samples:
        with Image.open(sample) as image:
            image.verify()

    assert (deploy / "SHA256SUMS").is_file()
    assert (ROOT / "docs" / "DEPLOYMENT.md").is_file()
    assert (ROOT / "results" / "accuracy_summary.png").is_file()
    assert (ROOT / "results" / "plantdoc_domain_gap.png").is_file()
