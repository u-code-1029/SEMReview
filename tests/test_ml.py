import hashlib
import itertools
import json
import zipfile

import numpy as np
import pytest
import tifffile
from safetensors.numpy import save as save_tensors

from semreview import ml
from semreview.classdefs import (DEFAULT_CLASSES, DEFAULT_CLASS_DEFINITIONS, class_ids,
                                legacy_definitions, new_definition, normalize_model_metadata)


def test_ordered_decode_matches_brute_force():
    rng = np.random.default_rng(9)
    for count in range(3, 10):
        logits = rng.normal(size=(count, 3))
        best = max(itertools.combinations(range(count), 3), key=lambda p: sum(logits[p[c], c] for c in range(3)))
        assert ml.ordered_decode(logits) == list(best)
    assert ml.ordered_decode(np.zeros((5, 3))) == [0, 1, 2]
    with pytest.raises(ValueError):
        ml.ordered_decode(np.zeros((2, 3)))


def test_group_folds_do_not_leak_acquisition_parents():
    sequences = [{"relative_path": f"{group}/sample{sample}"} for group in "abcd" for sample in range(3)]
    folds = ml.group_folds(sequences)
    assert len(folds) == 3
    validations = []
    for training, validation in folds:
        train_groups = {sequences[i]["relative_path"].split("/")[0] for i in training}
        val_groups = {sequences[i]["relative_path"].split("/")[0] for i in validation}
        assert not train_groups & val_groups
        assert set(training) | set(validation) == set(range(len(sequences)))
        validations.extend(validation)
    assert sorted(validations) == list(range(len(sequences)))
    assert ml.group_folds(sequences[:6]) == []
    assert ml.group_folds([{"relative_path": f"sample{i}"} for i in range(4)]) == []


def _package(tmp_path, *, extra=None, identifier="safe-model", corrupt=False, metadata_overrides=None):
    tensors = {"head.weight": np.zeros((13, 3), np.float32), "head.mean": np.zeros(13, np.float32),
               "head.std": np.ones(13, np.float32), "encoder.fake": np.ones(1, np.float32)}
    payload = save_tensors(tensors)
    metadata = {"format_version": 1, "architecture": "resnet18-frozen", "id": identifier,
                "preprocess_version": ml.PREPROCESS_VERSION, "classes": ["바닥", "천장1", "천장2"],
                "metrics": {"validated": False}, "weights_sha256": hashlib.sha256(payload).hexdigest()}
    metadata.update(metadata_overrides or {})
    path = tmp_path / "test.sem-model"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("weights.safetensors", payload + (b"broken" if corrupt else b""))
        if extra:
            archive.writestr(extra, b"malicious")
    return path


def test_model_package_safe_roundtrip_and_import(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    path = _package(tmp_path)
    info = ml.inspect_model(path)
    assert info["classes"] == class_ids(legacy_definitions(["바닥", "천장1", "천장2"]))
    assert [item["name"] for item in info["class_definitions"]] == ["바닥", "천장1", "천장2"]
    imported = ml.import_model(path, tmp_path / "project")
    assert imported["id"] == info["id"]
    assert imported["path"] != str(path)


@pytest.mark.parametrize("kwargs", [{"extra": "../escape"}, {"extra": "metadata.json"},
                                     {"identifier": "../../escape"}, {"corrupt": True}])
def test_reject_malformed_model_archive(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ml.inspect_model(_package(tmp_path, **kwargs))
    assert not (tmp_path.parent / "escape").exists()


@pytest.mark.parametrize("metadata", [
    {"name": {}}, {"created_at": None},
    {"metrics": {"validated": "yes"}},
    {"metrics": {"by_class": []}},
    {"metrics": {"by_class": {"바닥": []}}},
    {"metrics": {"by_class": {"바닥": {"mae": "1.0"}}}},
    {"metrics": {"by_class": {"바닥": {"mae": True}}}},
    {"metrics": {"by_class": {"바닥": {"mae": float("inf")}}}},
    {"metrics": {"folders_needing_correction": None}},
    {"metrics": {"reason": []}},
    {"optional_extension": {"not_a_number": float("nan")}},
])
def test_reject_display_metadata_before_import(tmp_path, monkeypatch, metadata):
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    with pytest.raises(ValueError):
        ml.import_model(_package(tmp_path, metadata_overrides=metadata), tmp_path / "project")
    assert not (tmp_path / "project" / "models").exists()


def test_valid_optional_display_metadata_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    metadata = {"name": "검증 모델", "created_at": "2026-10-15T00:00:00+00:00",
                "metrics": {"validated": True, "folds": 3, "folders": 3, "offset": -1,
                            "by_class": {"바닥": {"mae": 1.25, "exact": .5, "within_1": .8, "within_3": 1}},
                            "folders_needing_correction": .5, "reason": "", "end_class": "천장2"}}
    result = ml.inspect_model(_package(tmp_path, metadata_overrides=metadata))
    assert result["name"] == metadata["name"]
    assert result["metrics"] == normalize_model_metadata({**metadata, "classes": ["바닥", "천장1", "천장2"]})["metrics"]


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        self.identity = "fake-test-only"
        self.state = {"fake": np.ones(1, dtype=np.float32)}

    def extract(self, image, bounds, cancelled=None):
        return np.array([image.mean() / 255, image.std() / 255], dtype=np.float32)


def _sequences(tmp_path, groups=3):
    result, annotations = [], {}
    rng = np.random.default_rng(4)
    for group in range(groups):
        directory = tmp_path / f"group{group}" / "sample"
        directory.mkdir(parents=True)
        sequence = {"relative_path": f"group{group}/sample", "path": str(directory), "images": [], "issues": []}
        for i in range(9):
            path = directory / f"{i + 5}.tif"
            array = (rng.integers(0, 20, (16, 24)) + 25 * i).astype(np.uint8)
            tifffile.imwrite(path, array)
            sequence["images"].append({"path": str(path), "name": path.name, "number": i + 5,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        result.append(sequence)
        annotations[sequence["relative_path"]] = {"바닥": 1, "천장1": 4, "천장2": 6}
    return result, annotations


def test_train_infer_portable_package_and_fold_metrics(tmp_path, monkeypatch):
    sequences, annotations = _sequences(tmp_path)
    monkeypatch.setattr(ml, "_load_encoder", lambda *a, **k: FakeEncoder())
    monkeypatch.setattr(ml, "_Encoder", FakeEncoder)
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    result = ml.train(str(tmp_path / "project"), sequences, annotations, ["바닥", "천장1", "천장2"], "테스트", {"epochs": 8})
    assert result["metrics"]["validated"]
    assert result["metrics"]["folds"] == 3
    assert set(result["metrics"]["by_class"]) == set(class_ids(legacy_definitions(["바닥", "천장1", "천장2"])))
    predictions = ml.infer(result["path"], sequences, str(tmp_path / "other-project"))
    assert len(predictions) == 3
    for prediction in predictions:
        values = list(prediction["landmarks"].values())
        assert len(values) == 3
        assert all(a < b for a, b in zip(values, values[1:]))
        assert prediction["warnings"]
        assert 0 <= prediction["priority"] <= 1


def test_reject_stale_scan_and_never_silently_train_random_encoder(tmp_path):
    sequences, _ = _sequences(tmp_path, 1)
    path = sequences[0]["images"][0]["path"]
    tifffile.imwrite(path, np.zeros((16, 24), np.uint8))
    with pytest.raises(ValueError, match="변경"):
        ml._sequence_features(sequences[0], FakeEncoder(), tmp_path / "project")
    with pytest.raises(ValueError, match="다운로드"):
        ml._load_encoder(tmp_path / "project", {"allow_download": False})


def test_cache_reuses_features_but_checks_contents(tmp_path):
    sequences, _ = _sequences(tmp_path, 1)
    original, _, signature = ml._sequence_features(sequences[0], FakeEncoder(), tmp_path / "project")
    encoder = FakeEncoder()
    encoder.extract = lambda *args: pytest.fail("Should reuse feature cache")
    cached, _, other = ml._sequence_features(sequences[0], encoder, tmp_path / "project")
    assert signature == other
    np.testing.assert_array_equal(original, cached)


def test_inference_invalid_tiff_is_reported_without_landmarks(tmp_path, monkeypatch):
    path = _package(tmp_path)
    monkeypatch.setattr(ml, "_Encoder", FakeEncoder)
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"broken")
    sequence = {"relative_path": "group/failed", "images": [{"path": str(bad), "name": bad.name}], "issues": []}
    result = ml.infer(str(path), [sequence], str(tmp_path / "project"))[0]
    assert not result["landmarks"]
    assert result["priority"] == 1
    assert result["warnings"]


def test_invalid_and_unordered_labels_are_rejected(tmp_path):
    sequences, annotations = _sequences(tmp_path, 1)
    annotations[sequences[0]["relative_path"]]["천장1"] = 0
    with pytest.raises(ValueError, match="순서"):
        ml._validate_labels(sequences, annotations, ["바닥", "천장1", "천장2"])


def test_metrics_count_lost_contiguous_images():
    result = ml._metrics([[3, 6, 8]], [[1, 5, 7]], [20], ["floor", "ceiling1", "ceiling2"], {"offset": 0})
    assert result["missed_images"] == 2
    assert result["extra_images"] == 1
    assert result["folders_needing_correction"] == 1


def test_real_resnet_forward_and_scalar_serialization(tmp_path, monkeypatch):
    """Random weights are used only in this architecture smoke test, never in train."""
    import torch
    from torchvision.models import resnet18
    torch.set_num_threads(2)
    network = resnet18(weights=None)
    encoder = ml._Encoder(network.state_dict(), device="cpu", tile_batch_size=1)
    feature = encoder._forward([np.ones((64, 64), dtype=np.float32) * .4])
    assert feature.shape == (1, 512)
    assert np.isfinite(feature).all()
    tensors = {f"encoder.{key}": value for key, value in encoder.state.items()}
    tensors.update({"head.weight": np.zeros((ml.HEAD_FEATURES, 3), np.float32),
                    "head.mean": np.zeros(ml.HEAD_FEATURES, np.float32),
                    "head.std": np.ones(ml.HEAD_FEATURES, np.float32)})
    ml._validate_architecture(tensors, 3)
    tensors["head.weight"] = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError, match="구조"):
        ml._validate_architecture(tensors, 3)
    # Exercise the real train writer: BatchNorm scalar counters must stay 0-D.
    sequences, annotations = _sequences(tmp_path, 1)
    monkeypatch.setattr(ml, "_load_encoder", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(encoder, "extract", lambda image, bounds, cancelled=None:
                        np.tile(np.array([image.mean() / 255, image.std() / 255], np.float32), 512))
    result = ml.train(str(tmp_path / "project"), sequences, annotations,
                      ["바닥", "천장1", "천장2"], "serialization-test", {"epochs": 1})
    assert ml.inspect_model(result["path"])["id"] == result["id"]
    _, loaded = ml._read_package(result["path"])
    assert loaded["encoder.bn1.num_batches_tracked"].shape == ()


def test_v2_package_roundtrip_and_reject_mismatched_definitions(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    metadata = {"format_version": 2, "classes": DEFAULT_CLASSES, "class_definitions": DEFAULT_CLASS_DEFINITIONS}
    loaded = ml.inspect_model(_package(tmp_path, metadata_overrides=metadata))
    assert loaded["classes"] == DEFAULT_CLASSES
    assert loaded["class_definitions"] == DEFAULT_CLASS_DEFINITIONS
    for malformed in [{"class_definitions": []}, {"classes": list(reversed(DEFAULT_CLASSES))},
                      {"class_definitions": None}]:
        with pytest.raises(ValueError):
            ml.inspect_model(_package(tmp_path, metadata_overrides={**metadata, **malformed}))
    with pytest.raises(ValueError, match="정의가 없습니다"):
        ml.inspect_model(_package(tmp_path, metadata_overrides={"format_version": 2}))


def test_legacy_model_long_class_names_are_preserved_without_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    names = ["처음 나타나는 복잡한 패턴의 경계 " * 30, "중간 " * 50, "마지막 " * 50]
    names = [name.rstrip() for name in names]
    metadata = ml.inspect_model(_package(tmp_path, metadata_overrides={"classes": names}))
    assert [item["name"] for item in metadata["class_definitions"]] == names
    assert metadata["classes"] == class_ids(legacy_definitions(names))


def test_train_validates_definition_order_before_loading_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_load_encoder", lambda *args, **kwargs: pytest.fail("Must validate before loading encoder"))
    with pytest.raises(ValueError, match="순서"):
        ml.train(str(tmp_path), [], {}, list(reversed(DEFAULT_CLASSES)), "bad",
                 class_definitions=DEFAULT_CLASS_DEFINITIONS)


def test_twelve_class_training_and_inference_preserve_ids_and_snapshot(tmp_path, monkeypatch):
    definitions = [new_definition(f"전이 {i + 1}", i) for i in range(12)]
    definitions[10]["description"] = "열한 번째 경계가 처음 나타나는 이미지"
    ids = class_ids(definitions)
    sequences = [{"relative_path": "group/sample", "images": [{} for _ in range(20)]}]
    annotations = {"group/sample": dict(zip(ids, range(2, 14)))}
    features = np.arange(20 * 7, dtype=np.float32).reshape(20, 7) / 100
    rules = {"candidates": list(range(20)), "warnings": []}
    monkeypatch.setattr(ml, "_sequence_features", lambda *args, **kwargs: (features, rules, "data-sig"))
    monkeypatch.setattr(ml, "_load_encoder", lambda *args, **kwargs: FakeEncoder())
    monkeypatch.setattr(ml, "_Encoder", FakeEncoder)
    monkeypatch.setattr(ml, "_validate_architecture", lambda *args: None)
    result = ml.train(str(tmp_path), sequences, annotations, ids, "12 클래스", {"epochs": 1},
                      class_definitions=definitions)
    assert result["format_version"] == 2
    assert result["classes"] == ids
    assert result["class_definitions"] == definitions
    definitions[10]["name"] = "프로젝트에서 나중에 변경한 이름"
    inspected, tensors = ml._read_package(result["path"])
    assert inspected["class_definitions"][10]["name"] == "전이 11"
    assert tensors["head.weight"].shape == (7, 12)
    prediction = ml.infer(result["path"], sequences, str(tmp_path))[0]
    assert list(prediction["landmarks"]) == ids
    assert list(prediction["candidates"]) == ids
    positions = list(prediction["landmarks"].values())
    assert len(positions) == 12 and all(a < b for a, b in zip(positions, positions[1:]))
    assert not any(identifier in warning for identifier in ids for warning in prediction["warnings"])
