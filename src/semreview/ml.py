"""Frozen ResNet-18 features, ordered boundary learning and portable safe models.

No frame from a held-out acquisition group is used to fit its normalizer or
classifier. Model archives contain JSON and safetensors only, never pickle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from safetensors.numpy import load as load_tensors, save as save_tensors
from skimage.transform import resize

from .imaging import _check_cancel, _report, normalize, read_grayscale, rule_analysis

FORMAT_VERSION = 1
PREPROCESS_VERSION = "sem-folder-percentile-global768x512-six512-context3-v1"
WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHT_HASH_PREFIX = "f37072fd"
MAX_MODEL_BYTES = 256 * 1024 * 1024
HEAD_FEATURES = 512 * 2 * 5 + 3


def _validate_display_metadata(metadata: dict) -> None:
    """Reject malformed external metadata before it reaches project/UI storage."""
    for key in ("name", "created_at"):
        if key in metadata and not isinstance(metadata[key], str):
            raise ValueError(f"모델의 {key} 정보는 문자열이어야 합니다.")
    metrics = metadata["metrics"]
    if "validated" in metrics and not isinstance(metrics["validated"], bool):
        raise ValueError("모델의 검증 상태는 참 또는 거짓이어야 합니다.")

    def finite_number(value):
        try:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        except (OverflowError, TypeError):
            return False

    if "by_class" in metrics:
        by_class = metrics["by_class"]
        if not isinstance(by_class, dict):
            raise ValueError("클래스별 평가 정보는 객체여야 합니다.")
        for name, values in by_class.items():
            if not isinstance(name, str) or not isinstance(values, dict):
                raise ValueError("클래스별 평가 정보의 형식이 올바르지 않습니다.")
            if any(not finite_number(value) for value in values.values()):
                raise ValueError("클래스별 평가값은 유한한 숫자여야 합니다.")
    for key in ("folds", "folders", "missed_images", "extra_images", "folders_needing_correction", "offset"):
        if key in metrics and not finite_number(metrics[key]):
            raise ValueError(f"모델 평가값 {key}는 유한한 숫자여야 합니다.")
    for key in ("reason", "end_class"):
        if key in metrics and not isinstance(metrics[key], str):
            raise ValueError(f"모델 평가 정보 {key}는 문자열이어야 합니다.")
    # Python's JSON reader accepts nonstandard NaN/Infinity by default, whereas
    # project storage correctly forbids them, including in optional extensions.
    try:
        json.dumps(metadata, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("모델 메타데이터에 유효하지 않은 값이 있습니다.") from exc


def _validate_architecture(tensors: dict[str, np.ndarray], classes: int) -> None:
    """Validate the full fixed architecture without importing PyTorch in the UI."""
    shapes = {"conv1.weight": (64, 3, 7, 7)}

    def bn(prefix, channels):
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            shapes[f"{prefix}.{suffix}"] = (channels,)
        shapes[f"{prefix}.num_batches_tracked"] = ()

    bn("bn1", 64)
    previous = 64
    for stage, channels in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            prefix = f"layer{stage}.{block}"
            shapes[f"{prefix}.conv1.weight"] = (channels, previous, 3, 3)
            shapes[f"{prefix}.conv2.weight"] = (channels, channels, 3, 3)
            bn(f"{prefix}.bn1", channels)
            bn(f"{prefix}.bn2", channels)
            if stage > 1 and block == 0:
                shapes[f"{prefix}.downsample.0.weight"] = (channels, previous, 1, 1)
                bn(f"{prefix}.downsample.1", channels)
            previous = channels
    expected = {f"encoder.{key}": value for key, value in shapes.items()}
    expected.update({"head.weight": (HEAD_FEATURES, classes), "head.mean": (HEAD_FEATURES,), "head.std": (HEAD_FEATURES,)})
    if set(tensors) != set(expected) or any(tensors[key].shape != shape for key, shape in expected.items()):
        raise ValueError("ResNet-18 인코더 또는 경계 검출기 구조가 올바르지 않습니다.")
    for key, array in tensors.items():
        expected_dtype = np.dtype("int64") if key.endswith("num_batches_tracked") else np.dtype("float32")
        if array.dtype != expected_dtype:
            raise ValueError("모델 가중치의 숫자 형식이 올바르지 않습니다.")


def _sha(path: str | Path, cancelled=None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _check_cancel(cancelled)
            digest.update(chunk)
    return digest.hexdigest()


def _read_package(path) -> tuple[dict, dict[str, np.ndarray]]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 2 or set(names) != {"metadata.json", "weights.safetensors"}:
                raise ValueError("모델에는 metadata.json과 weights.safetensors만 있어야 합니다.")
            if any(info.file_size > MAX_MODEL_BYTES for info in archive.infolist()):
                raise ValueError("모델 파일이 허용 크기를 초과했습니다.")
            if archive.getinfo("metadata.json").file_size > 4 * 1024 * 1024:
                raise ValueError("모델 메타데이터가 너무 큽니다.")
            metadata = json.loads(archive.read("metadata.json"))
            payload = archive.read("weights.safetensors")
        if not isinstance(metadata, dict) or metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("지원하지 않는 모델 형식입니다.")
        if metadata.get("preprocess_version") != PREPROCESS_VERSION or metadata.get("architecture") != "resnet18-frozen":
            raise ValueError("지원하지 않는 모델 전처리 또는 인코더입니다.")
        classes = metadata.get("classes")
        if not isinstance(classes, list) or not 2 <= len(classes) <= 64 or any(not isinstance(c, str) or not c.strip() for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("모델의 클래스 정의가 올바르지 않습니다.")
        if hashlib.sha256(payload).hexdigest() != metadata.get("weights_sha256"):
            raise ValueError("모델 가중치의 체크섬이 일치하지 않습니다.")
        if not isinstance(metadata.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", metadata["id"]):
            raise ValueError("모델 식별자가 올바르지 않습니다.")
        if not isinstance(metadata.get("metrics"), dict):
            raise ValueError("모델 평가 정보가 없습니다.")
        _validate_display_metadata(metadata)
        tensors = load_tensors(payload)
        required = ["head.weight", "head.mean", "head.std"]
        if any(k not in tensors for k in required) or not any(k.startswith("encoder.") for k in tensors):
            raise ValueError("모델에 필수 가중치가 없습니다.")
        weight, mean, std = (tensors[k] for k in required)
        if weight.ndim != 2 or weight.shape[1] != len(classes) or mean.shape != (weight.shape[0],) or std.shape != mean.shape:
            raise ValueError("모델 가중치의 크기가 올바르지 않습니다.")
        if any(not np.all(np.isfinite(array)) for array in tensors.values()) or np.any(std <= 0):
            raise ValueError("모델에 유효하지 않은 숫자가 포함되어 있습니다.")
        _validate_architecture(tensors, len(classes))
        return metadata, tensors
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"모델을 읽을 수 없습니다: {exc}") from exc


def inspect_model(path: str | Path) -> dict:
    metadata, _ = _read_package(path)
    return {**metadata, "path": str(Path(path).resolve())}


def import_model(path: str | Path, project_dir: str | Path) -> dict:
    inspect_model(path)
    target_dir = Path(project_dir) / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    # An untrusted model id is not used in a filesystem path.
    target = target_dir / f"import-{uuid.uuid4().hex}.sem-model"
    shutil.copyfile(path, target)
    return inspect_model(target)


def _device(requested="auto"):
    import torch
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("장치는 auto, cpu, cuda, mps 중 하나여야 합니다.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다. CPU 또는 자동 장치를 선택하세요.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS를 사용할 수 없습니다. CPU 또는 자동 장치를 선택하세요.")
    return torch.device(requested)


class _Encoder:
    def __init__(self, state, device="auto", tile_batch_size=2):
        import torch
        from torchvision.models import resnet18
        self.device = _device(device)
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Identity()
        state = {k: torch.as_tensor(v) for k, v in state.items() if not k.startswith("fc.")}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.tile_batch_size = max(1, min(int(tile_batch_size), 6))
        self.state = {k: v.detach().cpu().numpy().copy() for k, v in self.model.state_dict().items()}
        self.identity = hashlib.sha256(save_tensors(self.state)).hexdigest()

    def _forward(self, views, cancelled=None):
        import torch
        result = []
        offset = 0
        batch = self.tile_batch_size
        while offset < len(views):
            _check_cancel(cancelled)
            tensor = None
            try:
                array = np.stack(views[offset:offset + batch])[:, None, :, :]
                tensor = torch.from_numpy(array).repeat(1, 3, 1, 1).to(self.device)
                mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
                std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
                with torch.inference_mode():
                    output = self.model((tensor - mean) / std).cpu().numpy()
                result.extend(output)
                offset += len(output)
                del tensor, output
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                del tensor
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
                if batch > 1:
                    batch = max(1, batch // 2)
                    self.tile_batch_size = batch
                elif self.device.type != "cpu":
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                else:
                    raise RuntimeError("특징 추출 메모리가 부족합니다. 다른 프로그램을 닫고 다시 시도하세요.") from exc
        return np.asarray(result, dtype=np.float32)

    def extract(self, image, bounds, cancelled=None):
        image = normalize(image, bounds)
        global_view = resize(image, (512, 768), anti_aliasing=True, preserve_range=True).astype(np.float32)
        global_feature = self._forward([global_view], cancelled)[0]
        tiles = []
        rows = np.linspace(0, image.shape[0], 3, dtype=int)
        cols = np.linspace(0, image.shape[1], 4, dtype=int)
        for row in range(2):
            for col in range(3):
                tile = image[rows[row]:rows[row + 1], cols[col]:cols[col + 1]]
                tiles.append(resize(tile, (512, 512), anti_aliasing=True, preserve_range=True).astype(np.float32))
        tile_feature = self._forward(tiles, cancelled).mean(axis=0)
        return np.concatenate([global_feature, tile_feature]).astype(np.float32)


def _load_encoder(project_dir, config, progress=None, cancelled=None):
    import torch
    local = config.get("pretrained_weights_path")
    if local and str(local).endswith(".sem-model"):
        _, tensors = _read_package(local)
        state = {k.removeprefix("encoder."): v for k, v in tensors.items() if k.startswith("encoder.")}
        return _Encoder(state, config.get("device", "auto"), config.get("tile_batch_size", 2))
    cache = Path(project_dir) / "cache" / "encoders"
    cache.mkdir(parents=True, exist_ok=True)
    path = Path(local) if local else cache / "resnet18-f37072fd.pth"
    if not path.is_file():
        if local or not config.get("allow_download", False):
            raise ValueError("ImageNet 사전학습 가중치가 없습니다. 최초 다운로드를 허용하거나 공식 resnet18-f37072fd.pth 또는 .sem-model을 지정하세요.")
        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(WEIGHT_URL, timeout=30) as response, open(temporary, "wb") as output:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    _check_cancel(cancelled)
                    output.write(chunk)
                    downloaded += len(chunk)
                    _report(progress, "download", downloaded, total, "공식 ResNet-18 가중치 다운로드")
            if not _sha(temporary, cancelled).startswith(WEIGHT_HASH_PREFIX):
                raise ValueError("다운로드한 가중치의 체크섬이 일치하지 않습니다.")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if not _sha(path, cancelled).startswith(WEIGHT_HASH_PREFIX):
        raise ValueError("공식 ResNet-18 ImageNet 가중치의 체크섬과 일치하지 않습니다.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return _Encoder(state, config.get("device", "auto"), config.get("tile_batch_size", 2))


def _context_features(base, scores):
    """Current frame, means of preceding/following 3, and their differences."""
    count = len(base)
    before = np.asarray([base[max(0, i - 3):i].mean(axis=0) if i else base[0] for i in range(count)])
    after = np.asarray([base[i + 1:min(count, i + 4)].mean(axis=0) if i + 1 < count else base[-1] for i in range(count)])
    rules = np.asarray(scores, dtype=np.float32)
    rule_context = np.stack([rules, np.roll(rules, 1), np.roll(rules, -1)], axis=1)
    rule_context[0, 1] = 0
    rule_context[-1, 2] = 0
    return np.concatenate([base, before, after, base - before, after - base, rule_context], axis=1).astype(np.float32)


def _sequence_features(sequence, encoder, project_dir, progress=None, cancelled=None):
    images = sequence["images"]
    if not images:
        raise ValueError("이미지가 없는 폴더는 학습할 수 없습니다.")
    # Always rehash contents: size and mtime alone do not protect cache identity.
    hashes = []
    for index, item in enumerate(images):
        digest = _sha(item["path"], cancelled)
        if item.get("sha256") and digest != item["sha256"]:
            raise ValueError(f"스캔 후 이미지가 변경되었습니다. 다시 스캔하고 라벨을 확인하세요: {item['name']}")
        hashes.append(digest)
        _report(progress, "fingerprint", index + 1, len(images), item["name"])
    signature = hashlib.sha256(json.dumps([PREPROCESS_VERSION, encoder.identity, hashes]).encode()).hexdigest()
    directory = Path(project_dir) / "cache" / "features"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{signature}.npz"
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as data:
                features = data["features"].copy()
                rules = json.loads(str(data["rules"].item()))
                if features.shape[0] == len(images) and np.all(np.isfinite(features)):
                    return features, rules, signature
        except (ValueError, OSError, KeyError):
            pass
    rules = rule_analysis(sequence, progress, cancelled)
    bounds = tuple(rules["normalization_bounds"])
    base = []
    for index, item in enumerate(images):
        _check_cancel(cancelled)
        base.append(encoder.extract(read_grayscale(item["path"]), bounds, cancelled))
        _report(progress, "features", index + 1, len(images), f"{sequence['relative_path']} / {item['name']}")
    features = _context_features(np.asarray(base), rules["scores"])
    if any(_sha(item["path"], cancelled) != digest for item, digest in zip(images, hashes)):
        raise ValueError("특징 추출 중 이미지가 변경되었습니다. 다시 스캔하세요.")
    temp = directory / f"{signature}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp, "wb") as output:
            np.savez_compressed(output, features=features, rules=json.dumps(rules, ensure_ascii=False))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return features, rules, signature


def ordered_decode(logits: np.ndarray) -> list[int]:
    """Maximum-sum strictly increasing positions, O(frames * classes)."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.all(np.isfinite(logits)):
        raise ValueError("경계 점수가 올바르지 않습니다.")
    count, classes = logits.shape
    if classes < 1 or count < classes:
        raise ValueError("클래스 수보다 이미지가 적어 경계를 정할 수 없습니다.")
    previous = logits[:, 0].copy()
    parents = np.full((count, classes), -1, dtype=np.int64)
    for cls in range(1, classes):
        current = np.full(count, -np.inf)
        best_score, best_index = -np.inf, -1
        for index in range(count):
            if index > 0 and previous[index - 1] > best_score:
                best_score, best_index = previous[index - 1], index - 1
            if best_index >= 0:
                current[index] = best_score + logits[index, cls]
                parents[index, cls] = best_index
        previous = current
    index = int(np.argmax(previous))
    positions = [index]
    for cls in range(classes - 1, 0, -1):
        index = int(parents[index, cls])
        positions.append(index)
    return positions[::-1]


def group_folds(sequences: list[dict], level: int = 1) -> list[tuple[list[int], list[int]]]:
    """Use ancestor paths as groups; never fall back to splitting image frames."""
    level = max(1, int(level))
    groups = {}
    for index, sequence in enumerate(sequences):
        parts = PurePosixPath(sequence["relative_path"].replace("\\", "/")).parts
        # A sequence directory itself cannot be an acquisition parent.
        parent_parts = parts[:-1]
        group = "/".join(parent_parts[:level]) if parent_parts else "<base>"
        groups.setdefault(group, []).append(index)
    if len(groups) < 3:
        return []
    buckets = [[] for _ in range(3)]
    for group in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        bucket = min(range(3), key=lambda i: len(buckets[i]))
        buckets[bucket].extend(groups[group])
    all_indices = set(range(len(sequences)))
    return [(sorted(all_indices - set(validation)), sorted(validation)) for validation in buckets]


def _fit(features, labels, classes, config, progress=None, cancelled=None):
    import torch
    torch.manual_seed(42)
    # The small linear head stays on CPU so accelerator memory is for images.
    matrix = np.concatenate(features).astype(np.float32)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    std = np.maximum(std, 0.05).astype(np.float32)
    inputs = torch.from_numpy(np.clip((matrix - mean) / std, -12, 12))
    weight = torch.zeros((matrix.shape[1], classes), requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=float(config.get("learning_rate", 0.015)))
    offsets = np.cumsum([0] + [len(f) for f in features])
    targets = [torch.tensor(y, dtype=torch.long) for y in labels]
    epochs = max(1, min(int(config.get("epochs", 120)), 2000))
    regularization = max(0.0, float(config.get("regularization", .05)))
    for epoch in range(epochs):
        _check_cancel(cancelled)
        optimizer.zero_grad(set_to_none=True)
        logits = inputs @ weight
        losses = [torch.nn.functional.cross_entropy(logits[offsets[i]:offsets[i + 1]].T, targets[i]) for i in range(len(features))]
        loss = torch.stack(losses).mean() + regularization * weight.square().sum() / classes
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch + 1 == epochs:
            _report(progress, "fit", epoch + 1, epochs, f"경계 검출기 학습 · 손실 {loss.item():.4f}")
    return {"head.weight": weight.detach().numpy(), "head.mean": mean, "head.std": std}


def _logits(features, head):
    return np.clip((features - head["head.mean"]) / head["head.std"], -12, 12) @ head["head.weight"]


def _metrics(predictions, targets, lengths, classes, config):
    errors = np.abs(np.asarray(predictions) - np.asarray(targets))
    by_class = {name: {"mae": float(errors[:, i].mean()), "exact": float((errors[:, i] == 0).mean()),
                       "within_1": float((errors[:, i] <= 1).mean()), "within_3": float((errors[:, i] <= 3).mean())}
                for i, name in enumerate(classes)}
    end_class = config.get("end_class", classes[-1])
    end_index = classes.index(end_class) if end_class in classes else len(classes) - 1
    offset = int(config.get("offset", 30))
    missed, extra = [], []
    for predicted, target, length in zip(predictions, targets, lengths):
        pred_start, pred_end = predicted[0], min(length - 1, predicted[end_index] + offset)
        true_start, true_end = target[0], min(length - 1, target[end_index] + offset)
        overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start) + 1)
        missed.append(max(0, true_end - true_start + 1) - overlap)
        extra.append(max(0, pred_end - pred_start + 1) - overlap)
    return {"validated": True, "folds": 3, "folders": len(targets), "by_class": by_class,
            "missed_images": int(sum(missed)), "extra_images": int(sum(extra)),
            "folders_needing_correction": float(np.any(errors > 0, axis=1).mean()),
            "end_class": classes[end_index], "offset": offset}


def _validate_labels(sequences, annotations, classes):
    if not 2 <= len(classes) <= 64 or len(set(classes)) != len(classes) or any(not isinstance(c, str) or not c.strip() for c in classes):
        raise ValueError("서로 다른 이름의 클래스를 2~64개 지정하세요.")
    chosen, targets = [], []
    for sequence in sequences:
        labels = annotations.get(sequence["relative_path"], {})
        if not all(name in labels for name in classes):
            continue
        positions = [labels[name] for name in classes]
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(sequence["images"]) for i in positions):
            raise ValueError(f"라벨 위치가 올바르지 않습니다: {sequence['relative_path']}")
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError(f"라벨은 클래스 순서대로 증가해야 합니다: {sequence['relative_path']}")
        if sequence.get("issues"):
            raise ValueError(f"읽기 오류·번호 문제를 해결한 뒤 학습하세요: {sequence['relative_path']}")
        chosen.append(sequence)
        targets.append([int(i) for i in positions])
    if not chosen:
        raise ValueError("모든 클래스 라벨을 확정한 폴더가 하나 이상 필요합니다.")
    return chosen, targets


def train(project_dir: str, sequences: list[dict], annotations: dict, classes: list[str], model_name: str,
          config: dict | None = None, progress=None, cancelled=None) -> dict:
    config = dict(config or {})
    sequences, labels = _validate_labels(sequences, annotations, classes)
    _check_cancel(cancelled)
    encoder = _load_encoder(project_dir, config, progress, cancelled)
    features, signatures = [], []
    for index, sequence in enumerate(sequences):
        _report(progress, "folders", index, len(sequences), sequence["relative_path"])
        data, _, signature = _sequence_features(sequence, encoder, project_dir, progress, cancelled)
        features.append(data)
        signatures.append(signature)
    folds = group_folds(sequences, config.get("group_level", 1))
    metrics = {"validated": False, "reason": "독립 촬영 그룹이 3개 미만이라 일반화 성능을 검증하지 못했습니다.", "folders": len(sequences)}
    if folds:
        predicted, actual, lengths = [], [], []
        for fold, (training, validation) in enumerate(folds):
            _report(progress, "validation", fold, len(folds), f"그룹 교차검증 {fold + 1}/3")
            head = _fit([features[i] for i in training], [labels[i] for i in training], len(classes), config, progress, cancelled)
            for i in validation:
                predicted.append(ordered_decode(_logits(features[i], head)))
                actual.append(labels[i])
                lengths.append(len(features[i]))
        metrics = _metrics(predicted, actual, lengths, classes, config)
    head = _fit(features, labels, len(classes), config, progress, cancelled)
    metadata = {"format_version": FORMAT_VERSION, "preprocess_version": PREPROCESS_VERSION,
                "architecture": "resnet18-frozen", "id": uuid.uuid4().hex, "name": model_name.strip() or "SEM 모델",
                "created_at": datetime.now(timezone.utc).isoformat(), "classes": list(classes), "metrics": metrics,
                "encoder_sha256": encoder.identity,
                "training_config": {key: value for key, value in config.items() if key in {"epochs", "learning_rate", "regularization", "group_level", "end_class", "offset"}},
                "training_data": {sequence["relative_path"]: {"signature": signature, "labels": labels[i]}
                                  for i, (sequence, signature) in enumerate(zip(sequences, signatures))},
                "caveat": "경계는 추천 결과입니다. 클래스 누락 여부와 원본 보존 범위를 사용자가 확인해야 합니다."}
    tensors = {f"encoder.{k}": v for k, v in encoder.state.items()}
    tensors.update(head)
    # ascontiguousarray promotes scalar BatchNorm counters to shape (1,).
    # np.array preserves their zero-dimensional shape for portable roundtrips.
    payload = save_tensors({k: np.array(v, copy=True, order="C") for k, v in tensors.items()})
    metadata["weights_sha256"] = hashlib.sha256(payload).hexdigest()
    directory = Path(project_dir) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{metadata['id']}.sem-model"
    temporary = path.with_suffix(".tmp")
    _check_cancel(cancelled)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            archive.writestr("weights.safetensors", payload)
        _check_cancel(cancelled)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**metadata, "path": str(path.resolve())}


def infer(model_path: str, sequences: list[dict], project_dir: str, progress=None, cancelled=None) -> list[dict]:
    metadata, tensors = _read_package(model_path)
    state = {k.removeprefix("encoder."): v for k, v in tensors.items() if k.startswith("encoder.")}
    encoder = _Encoder(state)
    classes = metadata["classes"]
    results = []
    for index, sequence in enumerate(sequences):
        _check_cancel(cancelled)
        _report(progress, "folders", index, len(sequences), sequence["relative_path"])
        result = {"relative_path": sequence["relative_path"], "landmarks": {}, "candidates": {},
                  "priority": 1.0, "warnings": [], "model_id": metadata["id"]}
        try:
            if sequence.get("issues"):
                raise ValueError("폴더의 파일·번호 문제를 해결한 뒤 다시 분석하세요.")
            data, rules, _ = _sequence_features(sequence, encoder, project_dir, progress, cancelled)
            logits = _logits(data, tensors)
            positions = ordered_decode(logits)
            result["landmarks"] = dict(zip(classes, positions))
            margins, disagreements = [], []
            for cls, name in enumerate(classes):
                ranked = np.argsort(-logits[:, cls], kind="stable")
                result["candidates"][name] = [int(i) for i in ranked[:5]]
                margin = float(logits[ranked[0], cls] - logits[ranked[1], cls]) if len(ranked) > 1 else 0.0
                margins.append(margin)
                if positions[cls] != int(ranked[0]):
                    result["warnings"].append(f"{name}: 클래스 순서 제약 때문에 단일 최고 후보와 다른 위치를 추천했습니다.")
                if margin < 0.25:
                    result["warnings"].append(f"{name}: 후보 간 점수 차이가 작습니다.")
                if cls > 0 and rules["candidates"] and min(abs(positions[cls] - p) for p in rules["candidates"]) > 3:
                    disagreements.append(name)
            if disagreements:
                result["warnings"].append("규칙 변화 후보와 모델 경계가 다릅니다: " + ", ".join(disagreements))
            if rules["warnings"]:
                result["warnings"].append(f"인접 이미지 정합이 불안정한 구간 {len(rules['warnings'])}개")
            if not metadata["metrics"].get("validated", False):
                result["warnings"].append("이 모델은 독립 그룹 일반화 성능이 검증되지 않았습니다.")
            result["priority"] = float(min(1, math.exp(-max(0, min(margins))) + .15 * len(disagreements)))
            result["rule_analysis"] = rules
            result["warnings"].append("모든 클래스가 존재하는지 확인하세요. 순서 제약으로 선택된 경계는 확정 라벨이 아닙니다.")
        except (ValueError, OSError) as exc:
            result["warnings"].append(str(exc))
        results.append(result)
    return results
