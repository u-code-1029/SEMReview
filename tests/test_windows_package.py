"""Portable download packaging and Windows split-archive reconstruction."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "package_windows", Path(__file__).resolve().parents[1] / "scripts" / "package_windows.py",
)
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


@pytest.fixture
def app(tmp_path):
    source = tmp_path / "app"
    source.mkdir()
    (source / "SEMReview.exe").write_bytes(b"test executable")
    (source / "_internal").mkdir()
    (source / "_internal" / "runtime.dll").write_bytes(os.urandom(12_000))
    (source / "_internal" / "한글.txt").write_text("test", encoding="utf-8")
    return source


def test_normal_zip_contains_only_explicit_app_files(app, tmp_path):
    (tmp_path / "private-training.tif").write_bytes(b"never package this")
    assets = packaging.package_windows(app, tmp_path / "release", "cpu", "0.1.0")
    archive = next(path for path in assets if path.suffix == ".zip")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("SEMReview/SEMReview.exe") == b"test executable"
        assert bundle.read("SEMReview/_internal/한글.txt") == b"test"
        assert all("private-training" not in name for name in bundle.namelist())
        metadata = json.loads(bundle.read("SEMReview/BUILD-INFO.json"))
        assert metadata["variant"] == "cpu"
        assert metadata["gpu_hardware_validated"] is False
    checksums = next(path for path in assets if path.name.endswith("SHA256SUMS.txt"))
    for line in checksums.read_text().splitlines():
        digest, name = line.split("  ")
        assert packaging.sha256(checksums.parent / name) == digest


def test_split_zip_reassembles_with_manifest_checksums(app, tmp_path):
    assets = packaging.package_windows(app, tmp_path / "release", "nvidia-cu126", "0.1.0-rc.1", max_part_bytes=4096)
    manifest = json.loads(next(path for path in assets if path.name.endswith("manifest.json")).read_text())
    assert len(manifest["parts"]) >= 3
    payload = bytearray()
    for part in manifest["parts"]:
        contents = (assets[0].parent / part["name"]).read_bytes()
        assert 0 < len(contents) <= 4096
        assert len(contents) == part["bytes"]
        assert hashlib.sha256(contents).hexdigest() == part["sha256"]
        payload.extend(contents)
    assert len(payload) == manifest["bytes"]
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    assembled = tmp_path / "assembled.zip"
    assembled.write_bytes(payload)
    with zipfile.ZipFile(assembled) as bundle:
        assert bundle.read("SEMReview/_internal/runtime.dll") == (app / "_internal/runtime.dll").read_bytes()
    assert any(path.suffix == ".cmd" for path in assets)
    assert any(path.suffix == ".ps1" for path in assets)


def test_refuses_source_output_and_existing_assets(app, tmp_path):
    with pytest.raises(ValueError, match="inside"):
        packaging.package_windows(app, app / "release", "cpu", "0.1.0")
    packaging.package_windows(app, tmp_path / "release", "cpu", "0.1.0")
    with pytest.raises(FileExistsError):
        packaging.package_windows(app, tmp_path / "release", "cpu", "0.1.0")


def test_rejects_unsafe_version_and_oversize_part(app, tmp_path):
    with pytest.raises(ValueError, match="Version"):
        packaging.package_windows(app, tmp_path / "release", "cpu", "0.1.0/evil")
    with pytest.raises(ValueError, match="2 GiB"):
        packaging.package_windows(app, tmp_path / "release", "cpu", "0.1.0", max_part_bytes=2 * 1024**3)


def test_windows_helper_extracts_and_rejects_corrupt_part(app, tmp_path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is required to run the Windows extraction helper")
    assets = packaging.package_windows(app, tmp_path / "release", "cpu", "0.1.0", max_part_bytes=4096)
    helper = next(path for path in assets if path.suffix == ".ps1")
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(next(path for path in assets if path.name.endswith("manifest.json")).read_text())
    extracted = helper.parent / Path(manifest["archive"]).stem
    assert (extracted / "SEMReview/_internal/runtime.dll").read_bytes() == (app / "_internal/runtime.dll").read_bytes()
    shutil.rmtree(extracted)
    part = helper.parent / manifest["parts"][0]["name"]
    payload = bytearray(part.read_bytes())
    payload[0] ^= 255
    part.write_bytes(payload)
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Checksum failed" in result.stdout
    assert not extracted.exists()
