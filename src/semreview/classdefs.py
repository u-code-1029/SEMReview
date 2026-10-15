"""Stable class identities, editable presentation, and legacy boundary adapters."""
from __future__ import annotations

import copy
import re
import uuid

CLASS_COLORS = ["#62d6bb", "#79aef7", "#e4af66", "#cb94ed", "#f18095", "#90cb77", "#7ad2e7", "#dbbf88"]
LEGACY_DEFAULT_NAMES = ["바닥", "1층 천장", "2층 천장"]
DEFAULT_CLASS_DEFINITIONS = [
    {"id": key, "name": name, "description": "", "color": CLASS_COLORS[i]}
    for i, (key, name) in enumerate(zip(("start", "middle", "end"), ("시작 기준점", "중간 기준점", "종료 기준점")))
]
DEFAULT_CLASSES = [item["id"] for item in DEFAULT_CLASS_DEFINITIONS]
_LEGACY_NAMESPACE = uuid.UUID("e72d1b9d-9260-5f98-a29a-9c6491aa549b")


def validate_definitions(defs: list[dict]) -> list[dict]:
    """Return an independent, normalized definition list or reject ambiguity."""
    if not isinstance(defs, list) or not 2 <= len(defs) <= 64:
        raise ValueError("클래스를 2~64개 정의하세요.")
    result = []
    for index, item in enumerate(defs):
        if not isinstance(item, dict):
            raise ValueError("클래스 정의는 객체여야 합니다.")
        key, name = item.get("id"), item.get("name")
        description, color = item.get("description", ""), item.get("color", CLASS_COLORS[index % len(CLASS_COLORS)])
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key):
            raise ValueError("클래스 ID는 영문·숫자·밑줄·하이픈 1~80자여야 합니다.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("클래스 이름은 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(description, str) or len(description) > 4000:
            raise ValueError("클래스 설명은 4000자 이하 문자열이어야 합니다.")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("클래스 색상은 #RRGGBB 형식이어야 합니다.")
        result.append({"id": key, "name": name.strip(), "description": description.strip(), "color": color.lower()})
    if len({item["id"] for item in result}) != len(result) or len({item["name"] for item in result}) != len(result):
        raise ValueError("클래스 ID와 이름은 각각 중복될 수 없습니다.")
    return result


def new_definition(name: str = "", index: int = 0) -> dict:
    return {"id": uuid.uuid4().hex, "name": name or f"클래스 {index + 1}", "description": "",
            "color": CLASS_COLORS[index % len(CLASS_COLORS)]}


def legacy_definitions(names: list[str]) -> list[dict]:
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("이전 클래스 이름 목록이 올바르지 않습니다.")
    return validate_definitions([
        {"id": uuid.uuid5(_LEGACY_NAMESPACE, name).hex, "name": name, "description": "",
         "color": CLASS_COLORS[i % len(CLASS_COLORS)]}
        for i, name in enumerate(names)
    ])


def class_ids(defs: list[dict]) -> list[str]:
    return [item["id"] for item in defs]


def class_name(class_id: str, defs: list[dict]) -> str:
    return next((item["name"] for item in defs if item["id"] == class_id), class_id)


def definitions_for_metadata(metadata: dict) -> list[dict]:
    if not isinstance(metadata, dict):
        raise ValueError("모델 메타데이터는 객체여야 합니다.")
    if "class_definitions" in metadata:
        defs = validate_definitions(metadata["class_definitions"])
        if metadata.get("classes", class_ids(defs)) != class_ids(defs):
            raise ValueError("모델 클래스 ID의 순서와 클래스 정의가 일치하지 않습니다.")
        return defs
    if metadata.get("format_version") == 2:
        raise ValueError("모델에 클래스 정의가 없습니다.")
    # Unversioned runtime payloads/fixtures may use the fresh project's IDs.
    if metadata.get("format_version") != 1 and metadata.get("classes") == DEFAULT_CLASSES:
        return validate_definitions(DEFAULT_CLASS_DEFINITIONS)
    return legacy_definitions(metadata.get("classes"))


def remap_keys(values: dict, mapping: dict[str, str]) -> dict:
    """Translate known old keys, retaining unknown historical data verbatim."""
    if not isinstance(values, dict):
        raise ValueError("클래스별 값은 객체여야 합니다.")
    result = {}
    for key, value in values.items():
        translated = mapping.get(key, key)
        if translated in result:
            raise ValueError("클래스 키 변환 중 중복이 발생했습니다.")
        result[translated] = copy.deepcopy(value)
    return result


def normalize_model_metadata(metadata: dict) -> dict:
    """Normalize legacy names once; keep trained model order and identity fixed."""
    result = copy.deepcopy(metadata)
    defs = definitions_for_metadata(result)
    mapping = dict(zip(result.get("classes", class_ids(defs)), class_ids(defs)))
    result["class_definitions"], result["classes"] = defs, class_ids(defs)
    for section in ("metrics", "training_config"):
        values = result.get(section)
        if isinstance(values, dict):
            if "end_class" in values and isinstance(values["end_class"], str):
                values["end_class"] = mapping.get(values["end_class"], values["end_class"])
            if isinstance(values.get("by_class"), dict):
                values["by_class"] = remap_keys(values["by_class"], mapping)
    return result
