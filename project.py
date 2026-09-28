"""Patient-separated CNN versus frozen ImageNet VGG16 experiment.

No fabricated results: training/evaluation outputs are created only by actual runs.
TensorFlow is imported lazily so data and metric tests can run independently.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import csv
import importlib.metadata
import json
import platform
import re
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

LABELS = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.0001
    split_seed: int = 42
    train_seed: int = 42
    bootstrap_repetitions: int = 1000
    threshold: float = 0.5
    # Choose a NEW ID for a different training seed before viewing any test results.
    experiment_id: str = "primary_seed42"

    def validate(self):
        if self.image_size != 224 or self.threshold != 0.5:
            raise ValueError("Primary protocol fixes 224 pixels and threshold 0.5.")
        if min(self.batch_size, self.epochs, self.patience,
               self.bootstrap_repetitions) < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid training or bootstrap configuration")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.experiment_id):
            raise ValueError("Use letters/numbers/underscore/hyphen in experiment ID")


def canonical_patient(value):
    value = str(value).strip()
    match = re.search(r"P_\d+", value)
    return match.group(0) if match else value


def read_manifest(path):
    """Relative image paths are resolved relative to the manifest CSV directory."""
    path = Path(path).resolve()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"patient_id", "image_path", "pathology", "official_split"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f"Need nonempty manifest with columns {sorted(required)}")
    for col in required:
        df[col] = df[col].str.strip()
        if (df[col] == "").any():
            raise ValueError(f"Blank {col} in manifest")
    df["patient_id"] = df.patient_id.map(canonical_patient)
    df["pathology"] = df.pathology.str.upper()
    df["official_split"] = df.official_split.str.lower()
    if not set(df.pathology).issubset(LABELS):
        raise ValueError("Unknown pathology; review labels instead of guessing")
    if set(df.official_split) != {"train", "test"}:
        raise ValueError("official_split must contain both train and test")
    def resolve(p):
        p = Path(p)
        return str((p if p.is_absolute() else path.parent / p).resolve())
    df["image_path"] = df.image_path.map(resolve)
    missing = df.loc[~df.image_path.map(lambda p: Path(p).is_file()), "image_path"]
    if len(missing):
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing.iloc[0]}")
    if df.image_path.duplicated().any():
        raise ValueError("Repeated file path; audit duplicate or conflicting lesions")
    df["label"] = df.pathology.map(LABELS).astype(int)
    # Stable within a project folder; retain image IDs in saved manifests.
    if "image_id" not in df:
        df["image_id"] = [hashlib.sha256((p + '|' + s).encode()).hexdigest()[:20]
                          for p, s in zip(df.patient_id, df.image_path)]
    if (df.image_id == "").any() or df.image_id.duplicated().any():
        raise ValueError("image_id must be nonblank and unique")
    return df.reset_index(drop=True)


def validate_separation(df, column="split"):
    if df.groupby("patient_id")[column].nunique().max() > 1:
        raise ValueError("Patient leakage: a patient occurs in more than one split")
    if "pixel_hash" in df:
        duplicates = df[df.pixel_hash.duplicated(keep=False)]
        if not duplicates.empty:
            raise ValueError("Identical prepared pixels found; audit duplicates before training")
    if "raw_hash" in df and df.raw_hash.duplicated().any():
        raise ValueError("Identical source file content found; audit duplicates")


def make_split(df, seed=42):
    """Keep official test; take fold zero of a fixed five-fold development split."""
    validate_separation(df, "official_split")
    dev = df[df.official_split == "train"].copy()
    test = df[df.official_split == "test"].copy()
    if dev.patient_id.nunique() < 5 or dev.groupby("label").patient_id.nunique().min() < 5:
        raise ValueError("Need at least five development patients per class")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    ti, vi = next(splitter.split(dev, dev.label, groups=dev.patient_id))
    train, val = dev.iloc[ti].copy(), dev.iloc[vi].copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"
    result = pd.concat([train, val, test], ignore_index=True)
    validate_separation(result)
    for name, part in result.groupby("split"):
        if set(part.label) != {0, 1}:
            raise ValueError(f"{name} lacks both classes. Review protocol before training.")
    return result


def read_grayscale(path):
    """Return float32 grayscale pixels; explicit DICOM MONOCHROME handling."""
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom
        from pydicom.pixels import apply_modality_lut
        ds = pydicom.dcmread(path)
        if str(getattr(ds, "PhotometricInterpretation", "")) not in {
                "MONOCHROME1", "MONOCHROME2"}:
            raise ValueError(f"Unsupported DICOM photometric interpretation: {path}")
        pixels = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
        if pixels.ndim != 2:
            raise ValueError(f"Expected a single grayscale frame: {path}")
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
    else:
        with Image.open(path) as im:
            if im.mode in {"L", "I", "F", "I;16", "I;16B", "I;16L"}:
                pixels = np.array(im, dtype=np.float32)
            elif im.mode in {"RGB", "RGBA"}:
                rgb = np.asarray(im.convert("RGB"))
                if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and
                        np.array_equal(rgb[..., 1], rgb[..., 2])):
                    raise ValueError(f"Image is not grayscale; check annotations: {path}")
                pixels = rgb[..., 0].astype(np.float32)
            else:
                raise ValueError(f"Unsupported image mode {im.mode}: {path}")
    if not np.isfinite(pixels).all() or min(pixels.shape) < 8:
        raise ValueError(f"Invalid image pixels: {path}")
    if np.unique(pixels).size <= 4:
        raise ValueError(f"Possible ROI mask or blank image, not a mammogram crop: {path}")
    return pixels


def prepare_image(path, size=224):
    pixels = read_grayscale(path)
    lo, hi = float(pixels.min()), float(pixels.max())
    if hi <= lo:
        raise ValueError("Constant image")
    # Fixed per-image operation: no learned statistics from validation or test.
    scaled = np.rint(255 * (pixels - lo) / (hi - lo)).astype(np.uint8)
    im = Image.fromarray(scaled)
    im = ImageOps.pad(im, (size, size), method=Image.Resampling.BILINEAR,
                      color=0, centering=(0.5, 0.5))
    return im, {"source_height": pixels.shape[0], "source_width": pixels.shape[1],
                "source_min": lo, "source_max": hi}


def prepare_data(manifest, output_dir, cfg):
    """Create immutable data audit and model inputs. Do not silently drop failures."""
    cfg.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "status.json", {"created_utc": utc_now(), "status": "preparing"})
    df = make_split(read_manifest(manifest), cfg.split_seed)
    images = output_dir / "images"
    images.mkdir()
    rows, failures = [], []
    for row in df.to_dict("records"):
        try:
            im, info = prepare_image(row["image_path"], cfg.image_size)
            dest = (images / f'{row["image_id"]}.png').resolve()
            im.save(dest)
            row.update(info)
            row["prepared_path"] = str(dest)
            row["raw_hash"] = sha256(row["image_path"])
            row["pixel_hash"] = hashlib.sha256(im.tobytes()).hexdigest()
            rows.append(row)
        except Exception as e:
            failures.append({"image_id": row["image_id"], "error": str(e)})
    pd.DataFrame(failures, columns=["image_id", "error"]).to_csv(
        output_dir / "conversion_failures.csv", index=False)
    if failures:
        raise ValueError(f"{len(failures)} conversion failures. See conversion_failures.csv")
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "audit.csv", index=False)
    validate_separation(result)
    counts = result.groupby(["split", "label"]).agg(
        images=("image_id", "count"), patients=("patient_id", "nunique"))
    counts.to_csv(output_dir / "counts.csv")
    for split, part in result.groupby("split"):
        part.to_csv(output_dir / f"{split}.csv", index=False)
    write_json(output_dir / "status.json", {"created_utc": utc_now(), "status": "prepared",
        "manifest_sha256": sha256(manifest), "split_seed": cfg.split_seed,
        "image_size": cfg.image_size, "data_verified": False,
        "preprocessing": "grayscale; per-image minmax; 8-bit; aspect-preserving pad"})
    return result


def verify_data(data_dir, reviewer_note):
    if not str(reviewer_note).strip():
        raise ValueError("Describe which images and metadata you checked")
    path = Path(data_dir) / "status.json"
    status = json.loads(path.read_text())
    if status.get("status") != "prepared":
        raise ValueError("Data preparation has not completed")
    status.update(data_verified=True, verification_utc=utc_now(),
                  reviewer_note=reviewer_note)
    write_json(path, status)


def tf_module(seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def image_dataset(df, cfg, training=False, vgg=False):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices(
        (df.prepared_path.to_numpy(), df.label.to_numpy(dtype=np.float32)))
    if training:
        ds = ds.shuffle(len(df), seed=cfg.train_seed, reshuffle_each_iteration=True)
    def decode(path, y):
        x = tf.io.decode_png(tf.io.read_file(path), channels=3)
        x = tf.cast(x, tf.float32)
        x.set_shape((cfg.image_size, cfg.image_size, 3))
        if vgg:
            x = tf.keras.applications.vgg16.preprocess_input(x)
        return x, tf.reshape(y, (1,))
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def build_cnn(cfg):
    import tensorflow as tf
    L = tf.keras.layers
    inputs = tf.keras.Input((cfg.image_size, cfg.image_size, 3))
    x = L.Rescaling(1.0 / 255)(inputs)
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = L.MaxPooling2D()(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu", name="last_conv")(x)
    x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.5)(x)
    return tf.keras.Model(inputs, L.Dense(1, activation="sigmoid")(x), name="scratch_cnn")


def build_vgg(cfg, weights="imagenet"):
    """The training pipeline ALWAYS calls this with ImageNet weights."""
    import tensorflow as tf
    base = tf.keras.applications.VGG16(include_top=False, weights=weights,
        input_shape=(cfg.image_size, cfg.image_size, 3), pooling="avg")
    base.trainable = False
    head_input = tf.keras.Input((512,))
    x = tf.keras.layers.Dense(128, activation="relu")(head_input)
    x = tf.keras.layers.Dropout(0.5)(x)
    head = tf.keras.Model(head_input, tf.keras.layers.Dense(1, activation="sigmoid")(x),
                          name="vgg_classifier")
    return base, head


def new_run(data_dir, runs_dir, cfg):
    cfg.validate()
    data_dir = Path(data_dir).resolve()
    data_status = json.loads((data_dir / "status.json").read_text())
    if not data_status.get("data_verified"):
        raise ValueError("Inspect images and metadata, then run verify_data first")
    if data_status['image_size'] != cfg.image_size or data_status['split_seed'] != cfg.split_seed:
        raise ValueError("Config does not match the prepared dataset")
    run = Path(runs_dir).resolve() / cfg.experiment_id
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", asdict(cfg))
    source_path = Path(__file__).resolve()
    (run / "project_source.py").write_bytes(source_path.read_bytes())
    split_hashes = {s: sha256(data_dir / f"{s}.csv") for s in ["train", "val", "test"]}
    write_json(run / "status.json", {"created_utc": utc_now(), "stage": "created",
        "data_dir": str(data_dir), "data_status": data_status,
        "split_hashes": split_hashes, "source_sha256": sha256(source_path),
        "dataset": "user-supplied CBIS-DDSM mass crops"})
    versions = {}
    for pkg in ["tensorflow", "tensorflow-cpu", "keras", "numpy", "pandas",
                "scikit-learn", "pydicom", "Pillow", "matplotlib"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed under this package name"
    write_json(run / "environment.json", {"python": platform.python_version(),
        "platform": platform.platform(), "packages": versions})
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                    if d.metadata.get("Name"))
    (run / "environment_freeze.txt").write_text('\n'.join(freeze), encoding="utf-8")
    return run


def load_run(run):
    run = Path(run)
    cfg = Config(**json.loads((run / "config.json").read_text()))
    status = json.loads((run / "status.json").read_text())
    for split, digest in status['split_hashes'].items():
        if sha256(Path(status['data_dir']) / f'{split}.csv') != digest:
            raise ValueError("Split file changed since run creation")
    if sha256(Path(__file__)) != status['source_sha256']:
        raise ValueError("Source code changed since run creation; use saved source or a new run")
    return cfg, status


def train_models(run):
    run = Path(run)
    cfg, status = load_run(run)
    if status['stage'] != 'created':
        raise ValueError("This run has already started; use a new experiment ID for a new run")
    tf = tf_module(cfg.train_seed)
    status.update(stage="training", training_started_utc=utc_now())
    write_json(run / "status.json", status)
    train = pd.read_csv(Path(status['data_dir']) / 'train.csv')
    val = pd.read_csv(Path(status['data_dir']) / 'val.csv')
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train.label)
    class_weights = {int(k): float(v) for k, v in enumerate(weights)}
    write_json(run / 'class_weights.json', class_weights)
    write_json(run / 'devices.json', {"devices": [str(d) for d in tf.config.list_physical_devices()]})
    timings = []
    for name in ['cnn', 'vgg16']:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(cfg.train_seed)
        folder = run / name
        folder.mkdir()
        model_start = time.perf_counter()
        started = utc_now()
        if name == 'cnn':
            model = build_cnn(cfg)
            train_ds = image_dataset(train, cfg, training=True)
            val_ds = image_dataset(val, cfg)
            feature_seconds = 0.0
        else:
            base, model = build_vgg(cfg, weights='imagenet')
            feature_start = time.perf_counter()
            train_features = base.predict(image_dataset(train, cfg, vgg=True), verbose=1)
            val_features = base.predict(image_dataset(val, cfg, vgg=True), verbose=1)
            feature_seconds = time.perf_counter() - feature_start
            # Cache only train/validation features. Test is evaluated later.
            np.savez_compressed(folder / 'development_features.npz',
                train=train_features, val=val_features,
                train_ids=train.image_id.to_numpy(dtype=str),
                val_ids=val.image_id.to_numpy(dtype=str))
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_features, train.label.to_numpy(dtype=np.float32).reshape(-1, 1)))
            train_ds = train_ds.shuffle(len(train), seed=cfg.train_seed).batch(cfg.batch_size)
            val_ds = tf.data.Dataset.from_tensor_slices(
                (val_features, val.label.to_numpy(dtype=np.float32).reshape(-1, 1))).batch(cfg.batch_size)
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
            loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')])
        summary = []
        model.summary(print_fn=lambda line: summary.append(line))
        (folder / 'trained_model_summary.txt').write_text('\n'.join(summary), encoding='utf-8')
        class EpochClock(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.start = time.perf_counter()
            def on_epoch_end(self, epoch, logs=None):
                # Keep string timestamps out of Keras numeric progress metrics.
                row = dict(epoch=epoch, **(logs or {}),
                           epoch_seconds=time.perf_counter() - self.start,
                           ended_utc=utc_now())
                path = folder / 'training_log.csv'
                exists = path.exists()
                with open(path, 'a', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
        callbacks = [EpochClock(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.patience,
                restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(folder / 'best.weights.h5'),
                monitor='val_loss', save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()]
        fit_start = time.perf_counter()
        history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs,
                            class_weight=class_weights, callbacks=callbacks, verbose=1)
        fit_seconds = time.perf_counter() - fit_start
        values = np.asarray(history.history['val_loss'], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite validation loss; do not evaluate failed training')
        model.load_weights(folder / 'best.weights.h5')
        if name == 'vgg16':
            inputs = tf.keras.Input((cfg.image_size, cfg.image_size, 3))
            full_model = tf.keras.Model(inputs, model(base(inputs, training=False)),
                                        name='frozen_imagenet_vgg16')
        else:
            full_model = model
        full_model.save(folder / 'model.keras')
        summary = []
        full_model.summary(print_fn=lambda line: summary.append(line), expand_nested=True)
        (folder / 'full_model_summary.txt').write_text('\n'.join(summary), encoding='utf-8')
        timings.append({"model": name, "started_utc": started, "ended_utc": utc_now(),
            "epochs": len(values), "best_epoch": int(np.argmin(values)) + 1,
            "feature_seconds": feature_seconds, "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - model_start,
            "total_parameters": full_model.count_params(),
            "trainable_parameters": int(sum(np.prod(w.shape) for w in full_model.trainable_weights)),
            "pretrained_weights": 'ImageNet' if name == 'vgg16' else 'none',
            "model_sha256": sha256(folder / 'model.keras')})
        pd.DataFrame(timings).to_csv(run / 'training_summary.csv', index=False)
    status.update(stage='trained', training_completed_utc=utc_now())
    write_json(run / 'status.json', status)
    return pd.DataFrame(timings)


def metric_values(y, score, threshold=0.5):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Scores must be finite values in [0,1]')
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if both else None,
        "average_precision": average_precision_score(y, score) if both else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def paired_bootstrap(predictions, repeats=1000, seed=42):
    """Paired patient-cluster bootstrap of image-level ROC-AUC and sensitivity."""
    grouped = [np.asarray(indices) for indices in predictions.groupby('patient_id').indices.values()]
    if not grouped:
        raise ValueError('No patients')
    rng = np.random.default_rng(seed)
    samples = {f'{m}_{metric}': [] for m in ['cnn', 'vgg16', 'difference']
               for metric in ['roc_auc', 'sensitivity']}
    for _ in range(repeats):
        selected = rng.integers(0, len(grouped), len(grouped))
        idx = np.concatenate([grouped[i] for i in selected])
        b = predictions.iloc[idx]
        if b.label.nunique() < 2:
            continue
        scores = {m: metric_values(b.label, b[f'{m}_score']) for m in ['cnn', 'vgg16']}
        for metric in ['roc_auc', 'sensitivity']:
            for m in ['cnn', 'vgg16']:
                samples[f'{m}_{metric}'].append(scores[m][metric])
            samples[f'difference_{metric}'].append(scores['vgg16'][metric] - scores['cnn'][metric])
    rows = []
    for key, values in samples.items():
        lower, upper = np.quantile(values, [0.025, 0.975]) if values else [None, None]
        rows.append({'quantity': key, 'lower_95': lower, 'upper_95': upper,
                     'valid_replicates': len(values), 'requested_replicates': repeats})
    return pd.DataFrame(rows)


def evaluate_models(run):
    run = Path(run)
    cfg, status = load_run(run)
    if status['stage'] != 'trained':
        raise ValueError('Evaluate only a fully trained, unevaluated run')
    destination = run / 'evaluation'
    destination.mkdir(exist_ok=False)
    tf = tf_module(cfg.train_seed)
    test = pd.read_csv(Path(status['data_dir']) / 'test.csv')
    train = pd.read_csv(Path(status['data_dir']) / 'train.csv')
    pred = test[['image_id', 'patient_id', 'label']].copy()
    for optional in ['breast_density', 'image_view']:
        if optional in test:
            pred[optional] = test[optional]
    timings = pd.read_csv(run / 'training_summary.csv').set_index('model')
    metrics = []
    for name in ['cnn', 'vgg16']:
        path = run / name / 'model.keras'
        if sha256(path) != timings.loc[name, 'model_sha256']:
            raise ValueError('Trained model file changed')
        model = tf.keras.models.load_model(path, compile=False)
        start = time.perf_counter()
        score = model.predict(image_dataset(test, cfg, vgg=(name == 'vgg16')), verbose=1).ravel()
        pred[f'{name}_score'] = score
        values = metric_values(test.label, score, cfg.threshold)
        metrics.append(dict(model=name, **values, threshold=cfg.threshold,
            images=len(test), patients=test.patient_id.nunique(),
            prediction_seconds=time.perf_counter()-start))
        del model
        tf.keras.backend.clear_session()
    majority = int(train.label.value_counts().idxmax())
    metrics.append(dict(model='training_majority_reference',
        **metric_values(test.label, np.full(len(test), majority)), threshold=cfg.threshold,
        images=len(test), patients=test.patient_id.nunique(), prediction_seconds=0))
    results = pd.DataFrame(metrics)
    results.to_csv(destination / 'measured_results.csv', index=False)
    pred.to_csv(destination / 'test_predictions.csv', index=False)
    paired_bootstrap(pred, cfg.bootstrap_repetitions, cfg.train_seed).to_csv(
        destination / 'paired_patient_bootstrap.csv', index=False)
    # Exploratory slices: report sample sizes and avoid claiming demographic fairness.
    slices = []
    for column in ['breast_density', 'image_view']:
        if column not in pred:
            continue
        for group, frame in pred.groupby(column, dropna=False):
            for name in ['cnn', 'vgg16']:
                slices.append(dict(group_column=column, group=str(group), model=name,
                    images=len(frame), patients=frame.patient_id.nunique(),
                    malignant_images=int(frame.label.sum()),
                    **metric_values(frame.label, frame[f'{name}_score'])))
    if slices:
        pd.DataFrame(slices).to_csv(destination / 'exploratory_subgroups.csv', index=False)
    status.update(stage='evaluated', evaluation_completed_utc=utc_now())
    write_json(run / 'status.json', status)
    make_figures(run)
    make_report(run)
    return results


def make_figures(run):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
    run = Path(run)
    ev = run / 'evaluation'
    for name in ['cnn', 'vgg16']:
        hist = pd.read_csv(run / name / 'training_log.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot(hist.epoch + 1, hist[metric], label='Training')
            ax.plot(hist.epoch + 1, hist[f'val_{metric}'], label='Validation')
            ax.set(xlabel='Epoch', ylabel=metric, title=f'{name}: {metric}')
            ax.legend()
        fig.tight_layout(); fig.savefig(ev / f'{name}_learning_curves.png', dpi=160); plt.close(fig)
    pred = pd.read_csv(ev / 'test_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ['cnn', 'vgg16']):
        ConfusionMatrixDisplay.from_predictions(pred.label, (pred[f'{name}_score'] >= .5).astype(int),
            display_labels=['Benign', 'Malignant'], ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(ev / 'confusion_matrices.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ['cnn', 'vgg16']:
        RocCurveDisplay.from_predictions(pred.label, pred[f'{name}_score'], name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(pred.label, pred[f'{name}_score'], name=name, ax=axes[1])
    axes[1].axhline(pred.label.mean(), linestyle='--', color='gray', label='Test prevalence')
    axes[1].legend(); fig.tight_layout()
    fig.savefig(ev / 'roc_and_precision_recall.png', dpi=160); plt.close(fig)


def make_report(run):
    run = Path(run)
    results = pd.read_csv(run / 'evaluation/measured_results.csv')
    times = pd.read_csv(run / 'training_summary.csv')
    pred = pd.read_csv(run / 'evaluation/test_predictions.csv')
    lines = ['# Measured experiment results', '', f'Generated UTC: {utc_now()}', '',
        'These values were calculated from the saved test predictions for this run.',
        'Prediction unit: lesion image. Splits are patient-separated; scores are not patient-level screening results.', '',
        '| Model | ROC AUC | Sensitivity | Specificity | F1 |', '| --- | --- | --- | --- | --- |']
    for r in results.to_dict('records'):
        lines.append(f"| {r['model']} | {r['roc_auc']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | {r['f1']:.4f} |")
    lines.extend(['', f'Test images: {len(pred)}. Unique test patients: {pred.patient_id.nunique()}.',
        f'Malignant images: {int(pred.label.sum())}; benign images: {int((pred.label == 0).sum())}.', '',
        'See paired_patient_bootstrap.csv for paired patient-cluster percentile intervals.',
        'These intervals describe test-sample uncertainty conditional on these fitted models; they do not measure training-seed uncertainty.', '',
        '## Training evidence', ''])
    for r in times.to_dict('records'):
        lines.append(f"- {r['model']}: {r['epochs']} epochs; best epoch {r['best_epoch']}; total {r['total_seconds']:.1f} seconds; trainable parameters {r['trainable_parameters']}.")
    lines.extend(['', '## Interpretation still to write', '',
        'Discuss false negatives and false positives, uncertainty, class imbalance, source-image conversion and the restricted dataset.',
        'Do not claim that the larger point estimate proves superiority or that this prototype is clinically validated.',
        'Stakeholder feedback is recorded separately; no responses are inferred from model scores.', ''])
    (run / 'evaluation/RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def predict_image(run, model_name, path):
    if model_name not in {'cnn', 'vgg16'}:
        raise ValueError('model_name must be cnn or vgg16')
    cfg, _ = load_run(run)
    tf = tf_module(cfg.train_seed)
    model = tf.keras.models.load_model(Path(run) / model_name / 'model.keras', compile=False)
    image, _ = prepare_image(path, cfg.image_size)
    x = np.repeat(np.asarray(image)[..., None], 3, axis=-1)[None].astype(np.float32)
    if model_name == 'vgg16':
        x = tf.keras.applications.vgg16.preprocess_input(x)
    score = float(model.predict(x, verbose=0)[0, 0])
    return {'model': model_name, 'predicted_class': 'Malignant' if score >= cfg.threshold else 'Benign',
            'malignant_score': score, 'threshold': cfg.threshold,
            'note': 'Research model score; not a calibrated clinical probability.'}


def summarise_feedback(csv_path, destination):
    df = pd.read_csv(csv_path, keep_default_na=False)
    destination = Path(destination)
    if df.empty:
        text = '# Stakeholder feedback status\n\nNo stakeholder responses have been collected.\n'
    else:
        if df.participant_id.duplicated().any() or (df.participant_id == '').any():
            raise ValueError('Use one anonymous ID per respondent')
        if not df.consent_to_use.astype(str).str.lower().isin(['yes']).all():
            raise ValueError('Resolve consent before including these responses')
        dates = pd.to_datetime(df['date_utc'], utc=True, errors='raise')
        if (dates > pd.Timestamp.now(tz='UTC')).any():
            raise ValueError('Response dates cannot be in the future')
        text = f'# Stakeholder feedback summary\n\nActual responses: {len(df)}.\n\n'
        for col in ['clarity', 'ease_of_use', 'limitations_understood']:
            values = pd.to_numeric(df[col], errors='raise')
            if not values.between(1, 5).all():
                raise ValueError('Ratings must be 1 to 5')
            text += f'- {col}: median {values.median():.1f}; range {values.min()} to {values.max()}; n={len(values)}.\n'
        text += '\nRoles represented: ' + ', '.join(sorted(set(df.role))) + '.\n'
        text += '\nReview free-text responses manually for themes; record resulting changes in change_log.csv.\n'
        text += 'Convenience feedback does not validate diagnostic accuracy. Non-clinicians cannot establish clinical usefulness.\n'
    destination.write_text(text, encoding='utf-8')
    return text

