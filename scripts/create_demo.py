"""Create clearly synthetic SEM-like TIFF fixtures, never real measurement data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semreview.core import scan  # noqa: E402
from semreview.classdefs import class_ids, legacy_definitions, LEGACY_DEFAULT_NAMES  # noqa: E402
from semreview.project import DEFAULT_PROFILES, Project, sequence_signature  # noqa: E402


def create_demo(destination: Path, width=768, height=512, folders=2, frames=24):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("데모 대상은 비어 있는 새 폴더여야 합니다.")
    rng = np.random.default_rng(42)
    base = destination / "images"
    yy, xx = np.mgrid[:height, :width]
    for batch in range(3):
        for sample in range(folders):
            leaf = base / f"batch-{batch + 1:02}" / f"sample-{sample + 1:02}"
            leaf.mkdir(parents=True)
            for frame in range(frames):
                state = 0 if frame < 3 else 1 if frame < 9 else 2 if frame < 16 else 3
                sx, sy = (frame * 3 + batch * 7) % 17, (frame * 2 + sample * 5) % 13
                pattern = np.sin((xx + sx) / (14 + batch)) * np.cos((yy + sy) / (17 + sample))
                if state == 0:
                    field = 8000 + 1000 * pattern
                elif state == 1:
                    field = 20000 + 9000 * pattern
                elif state == 2:
                    field = 32000 + 12000 * (pattern > .3)
                else:
                    field = 44000 + 8000 * np.sin((xx + yy + sx) / 8)
                image = np.clip(field + rng.normal(0, 500, field.shape), 0, 65535).astype(np.uint16)
                tifffile.imwrite(leaf / f"{frame + 1}.tif", image, compression="deflate", photometric="minisblack")
    sequences = scan(str(base), DEFAULT_PROFILES)
    project_dir = destination / "project"
    definitions = legacy_definitions(LEGACY_DEFAULT_NAMES)
    for definition in definitions:
        definition["description"] = f"합성 데이터에서 {definition['name']} 상태가 처음 나타나는 이미지"
    classes = class_ids(definitions)
    with Project(project_dir) as project:
        project.set("base", str(base.resolve()))
        project.set("profiles", DEFAULT_PROFILES)
        project.update_class_definitions(definitions)
        project.set("sequences", sequences)
        project.set("demo", True)
        for sequence in sequences:
            project.save_annotation(sequence["relative_path"], dict(zip(classes, [3, 9, 16])),
                                    sequence_signature(sequence), "complete", 3)
    return project_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실제 SEM 데이터가 아닌 합성 데모를 생성합니다.")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--full-resolution", action="store_true")
    args = parser.parse_args()
    project = create_demo(args.destination, 3072 if args.full_resolution else 768,
                          2048 if args.full_resolution else 512)
    print(f"합성 데모 프로젝트: {project.resolve()}")
    print(f'python -m semreview --project "{project.resolve()}"')
