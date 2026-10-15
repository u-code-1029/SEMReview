import numpy as np
import pytest
import tifffile

from semreview import imaging


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_tiff_preserves_pixels_and_preview_is_uint8(tmp_path, dtype):
    original = (np.arange(48 * 64).reshape(48, 64) * 7).astype(dtype)
    path = tmp_path / "10.tif"
    tifffile.imwrite(path, original)
    assert np.array_equal(imaging.read_grayscale(path), original)
    preview = imaging.display_image(path, max_size=32)
    assert preview.dtype == np.uint8
    assert preview.shape == (24, 32)
    assert np.array_equal(tifffile.imread(path), original)


def test_rejects_multipage_rgb_float_and_corrupt(tmp_path):
    for name, array in [("stack", np.zeros((3, 16, 16), dtype=np.uint8)),
                        ("rgb", np.zeros((16, 16, 3), dtype=np.uint8)),
                        ("float", np.zeros((16, 16), dtype=np.float32))]:
        path = tmp_path / f"{name}.tif"
        if name == "stack":
            tifffile.imwrite(path, array, photometric="minisblack")
        else:
            tifffile.imwrite(path, array)
        with pytest.raises(ValueError):
            imaging.read_grayscale(path)
    corrupt = tmp_path / "bad.tif"
    corrupt.write_bytes(b"not a TIFF")
    with pytest.raises(ValueError):
        imaging.read_grayscale(corrupt)


def test_shift_is_compensated_but_global_change_is_suggested(tmp_path):
    rng = np.random.default_rng(12)
    initial = rng.integers(0, 130, (128, 192), dtype=np.uint8)
    translated = np.roll(initial, (4, -7), axis=(0, 1))
    changed = (translated.astype(np.uint16) + 100).astype(np.uint8)
    images = []
    for i, array in enumerate([initial, translated, changed]):
        path = tmp_path / f"{i + 1}.tif"
        tifffile.imwrite(path, array)
        images.append({"path": str(path), "name": path.name})
    result = imaging.rule_analysis({"images": images})
    assert result["shifts"][1] == [-4.0, 7.0]
    assert result["scores"][1] < .005
    assert result["scores"][2] > .05
    assert result["candidates"][0] == 2
    assert result["normalization_bounds"][1] > result["normalization_bounds"][0]


def test_rule_analysis_cancels_before_reading():
    with pytest.raises(RuntimeError, match="취소"):
        imaging.rule_analysis({"images": [{"path": "missing.tif"}]}, cancelled=lambda: True)


def test_constant_image_remains_finite(tmp_path):
    path = tmp_path / "1.tif"
    tifffile.imwrite(path, np.full((24, 32), 100, dtype=np.uint16))
    result = imaging.rule_analysis({"images": [{"path": str(path), "name": "1.tif"}] * 3})
    assert np.isfinite(result["scores"]).all()
    assert result["candidates"] == []
    assert len(result["warnings"]) == 2


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_miniswhite_is_canonicalized_without_modifying_tiff(tmp_path, dtype):
    original = (np.arange(48 * 64).reshape(48, 64) * 13).astype(dtype)
    black = tmp_path / "black.tif"
    white = tmp_path / "white.tif"
    stored_white = np.iinfo(dtype).max - original
    tifffile.imwrite(black, original, photometric="minisblack")
    tifffile.imwrite(white, stored_white, photometric="miniswhite")
    np.testing.assert_array_equal(imaging.read_grayscale(black), imaging.read_grayscale(white))
    np.testing.assert_array_equal(imaging.display_image(black), imaging.display_image(white))
    np.testing.assert_array_equal(tifffile.imread(white), stored_white)


def test_minimum_size_shift_analysis_stays_valid(tmp_path):
    rng = np.random.default_rng(84)
    original = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    paths = [tmp_path / "1.tif", tmp_path / "2.tif"]
    tifffile.imwrite(paths[0], original)
    tifffile.imwrite(paths[1], np.roll(original, (1, 1), axis=(0, 1)))
    result = imaging.rule_analysis({"images": [{"path": str(p), "name": p.name} for p in paths]})
    assert np.isfinite(result["scores"]).all()
    assert result["shifts"][1] == [-1.0, -1.0]
