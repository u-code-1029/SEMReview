"""TIFF decoding and shift-aware, deterministic labelling assistance.

Indices throughout this module refer to the numerically ordered image list,
never to the number in a filename. Display contrast is deliberately independent
of the folder-wide normalization used for analysis and learning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
from scipy import ndimage
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from skimage.transform import resize


def _check_cancel(cancelled: Callable | None) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("작업이 취소되었습니다.")


def _report(progress, stage, completed, total, message=""):
    if progress:
        progress({"stage": stage, "completed": completed, "total": total, "message": message})


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read a single grayscale TIFF, canonicalizing MINISWHITE to black=zero.

    Only the returned array is transformed; original TIFF bytes are untouched.
    """
    try:
        with tifffile.TiffFile(path) as handle:
            if len(handle.pages) != 1:
                raise ValueError("다중 페이지 TIFF는 지원하지 않습니다.")
            page = handle.pages[0]
            if page.samplesperpixel != 1 or len(page.shape) != 2 or page.photometric not in (0, 1):
                raise ValueError("2차원 그레이스케일 TIFF만 지원합니다.")
            white_is_zero = page.photometric == 0
            image = page.asarray()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"TIFF를 읽을 수 없습니다: {Path(path).name}: {exc}") from exc
    if image.ndim != 2 or image.dtype.kind != "u" or image.dtype.itemsize not in (1, 2):
        raise ValueError("8비트 또는 16비트 정수 그레이스케일 TIFF만 지원합니다.")
    if min(image.shape) < 8:
        raise ValueError("이미지의 가로와 세로는 최소 8픽셀이어야 합니다.")
    if white_is_zero:
        image = np.iinfo(image.dtype).max - image
    return image


def normalize(image: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return np.clip((image.astype(np.float32) - low) / max(high - low, 1.0), 0, 1)


def folder_bounds(sequence: dict, cancelled=None) -> tuple[float, float]:
    """Estimate a single intensity transform across evenly sampled frames."""
    images = sequence["images"]
    if not images:
        raise ValueError("폴더에 이미지가 없습니다.")
    values = []
    for i in np.unique(np.linspace(0, len(images) - 1, min(9, len(images))).astype(int)):
        _check_cancel(cancelled)
        image = read_grayscale(images[int(i)]["path"])
        step = max(1, int(np.sqrt(image.size / 65536)))
        values.append(image[::step, ::step].ravel())
    low, high = np.percentile(np.concatenate(values), [0.5, 99.5])
    if high <= low:
        high = low + 1
    return float(low), float(high)


def _thumbnail(image: np.ndarray, max_size: int) -> np.ndarray:
    scale = min(1.0, max_size / max(image.shape))
    if scale == 1:
        return image.astype(np.float32, copy=False)
    return resize(image, tuple(max(8, round(n * scale)) for n in image.shape),
                  order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)


def display_image(path: str | Path, max_size: int | None = 1536) -> np.ndarray:
    image = read_grayscale(path)
    low, high = np.percentile(image[::4, ::4], [0.5, 99.5])
    result = normalize(image, (float(low), float(high)))
    if max_size:
        result = _thumbnail(result, max_size)
    return np.ascontiguousarray(np.rint(result * 255).astype(np.uint8))


def _overlap(shape, shift):
    slices = []
    for length, offset in zip(shape, shift):
        slices.append(slice(max(0, int(np.ceil(offset))), min(length, int(np.floor(length + offset)))))
    return tuple(slices)


def rule_analysis(sequence: dict, progress=None, cancelled=None) -> dict:
    """Suggest changes, not class identities; never change labels or files."""
    images = sequence["images"]
    if not images:
        return {"scores": [], "candidates": [], "shifts": [], "warnings": ["이미지가 없습니다."]}
    bounds = folder_bounds(sequence, cancelled)
    raw_scores = [0.0]
    shifts = [[0.0, 0.0]]
    warnings = []
    previous = None
    for index, item in enumerate(images):
        _check_cancel(cancelled)
        current = _thumbnail(normalize(read_grayscale(item["path"]), bounds), 512)
        if previous is not None:
            if current.shape != previous.shape:
                raise ValueError(f"폴더 안 이미지 크기가 서로 다릅니다: {item['name']}")
            # Constant or nearly constant fields cannot provide reliable registration.
            reliable = min(float(previous.std()), float(current.std())) > 0.005
            shift = np.zeros(2)
            if reliable:
                shift, error, _ = phase_cross_correlation(previous, current, upsample_factor=1, normalization=None)
                reliable = bool(np.isfinite(error) and error < 0.95 and np.all(np.abs(shift) < np.array(current.shape) / 4))
            if reliable:
                aligned = ndimage.shift(current, shift, order=1, mode="nearest", prefilter=False)
                region = _overlap(current.shape, shift)
                a, b = previous[region], aligned[region]
            else:
                shift = np.zeros(2)
                a, b = previous, current
                warnings.append(f"{item['name']}: 정합 신뢰도가 낮아 구조 변화 신호의 비중을 낮췄습니다.")
            pixel_change = float(np.mean(np.abs(a - b)))
            histogram_a = np.histogram(a, bins=32, range=(0, 1), density=False)[0] / a.size
            histogram_b = np.histogram(b, bins=32, range=(0, 1), density=False)[0] / b.size
            histogram_change = float(np.abs(histogram_a - histogram_b).sum() / 2)
            edge_change = abs(float(np.mean(np.hypot(ndimage.sobel(a, axis=0), ndimage.sobel(a, axis=1)))) -
                              float(np.mean(np.hypot(ndimage.sobel(b, axis=0), ndimage.sobel(b, axis=1))))) / 4
            window = min(7, *a.shape)
            window -= int(window % 2 == 0)
            structure_change = (float(1 - structural_similarity(a, b, data_range=1.0, win_size=window)) / 2
                                if window >= 3 else pixel_change)
            raw_scores.append(float(np.clip(0.35 * pixel_change + 0.25 * histogram_change +
                                           0.15 * edge_change + (0.25 if reliable else 0.08) * structure_change, 0, 1)))
            shifts.append([float(s) for s in shift])
        previous = current
        _report(progress, "rules", index + 1, len(images), item["name"])
    scores = np.asarray(raw_scores)
    # Suppress adjacent peaks so suggestions cover distinct transitions.
    candidates = []
    for index in np.argsort(-scores, kind="stable"):
        if index > 0 and scores[index] > 0 and all(abs(int(index) - selected) >= 3 for selected in candidates):
            candidates.append(int(index))
        if len(candidates) >= 12:
            break
    return {"scores": scores.tolist(), "candidates": candidates, "shifts": shifts,
            "warnings": warnings, "normalization_bounds": list(bounds)}
