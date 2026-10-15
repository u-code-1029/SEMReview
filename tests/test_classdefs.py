import copy

import pytest

from semreview.classdefs import (DEFAULT_CLASS_DEFINITIONS, DEFAULT_CLASSES, class_ids,
                                definitions_for_metadata, legacy_definitions, new_definition,
                                normalize_model_metadata, validate_definitions)


def test_definitions_are_independent_and_legacy_ids_stay_stable():
    defs = validate_definitions(DEFAULT_CLASS_DEFINITIONS)
    defs[0]["name"] = "변경"
    assert DEFAULT_CLASS_DEFINITIONS[0]["name"] == "시작 기준점"
    legacy = legacy_definitions(["초기", "경계", "완료"])
    assert class_ids(legacy) == class_ids(legacy_definitions(["초기", "경계", "완료"]))
    assert legacy[0]["id"] == legacy_definitions(["초기", "완료"])[0]["id"]
    assert new_definition("경계")["id"] != new_definition("경계")["id"]


@pytest.mark.parametrize("change", [
    lambda defs: defs[:1],
    lambda defs: defs * 22,
    lambda defs: [dict(item, id="duplicate") for item in defs],
    lambda defs: [dict(item, name="duplicate") for item in defs],
    lambda defs: [dict(item, name="  ") for item in defs],
    lambda defs: [dict(item, id="../unsafe") for item in defs],
    lambda defs: [dict(item, color="red") for item in defs],
    lambda defs: [dict(item, description=[]) for item in defs],
])
def test_invalid_class_definitions_are_rejected(change):
    with pytest.raises(ValueError):
        validate_definitions(change(copy.deepcopy(DEFAULT_CLASS_DEFINITIONS)))


def test_legacy_model_metadata_normalizes_references_without_mutation():
    original = {"format_version": 1, "classes": ["A", "B"],
                "metrics": {"by_class": {"A": {"mae": 1}}, "end_class": "B"},
                "training_config": {"end_class": "B"}}
    result = normalize_model_metadata(original)
    ids = class_ids(legacy_definitions(["A", "B"]))
    assert result["classes"] == ids
    assert result["metrics"]["by_class"] == {ids[0]: {"mae": 1}}
    assert result["metrics"]["end_class"] == result["training_config"]["end_class"] == ids[1]
    assert original["classes"] == ["A", "B"]
    assert normalize_model_metadata(result) == result


def test_v2_metadata_requires_definitions_and_matching_order():
    with pytest.raises(ValueError, match="정의가 없습니다"):
        definitions_for_metadata({"format_version": 2, "classes": DEFAULT_CLASSES})
    with pytest.raises(ValueError, match="순서"):
        definitions_for_metadata({"format_version": 2, "classes": list(reversed(DEFAULT_CLASSES)),
                                  "class_definitions": DEFAULT_CLASS_DEFINITIONS})
