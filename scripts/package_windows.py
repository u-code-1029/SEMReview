"""Create portable Windows release assets, splitting large ZIPs for GitHub Releases.

Only the explicitly supplied PyInstaller directory is packaged. Project databases,
training images, caches, and local model weights are never discovered automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


DEFAULT_PART_BYTES = 1800 * 1024 * 1024
VARIANTS = ("cpu", "nvidia-cu126")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows_instructions(variant: str) -> str:
    return f"""SEM Review - Windows x64 ({variant})

1. Download all files for your chosen variant to the same folder.
2. If there is one .zip file, extract the entire ZIP in File Explorer.
   If there are .zip.001, .zip.002, ... parts, also download the matching
   manifest.json, Extract-*.cmd and Extract-*.ps1. Double-click Extract-*.cmd.
   The helper verifies every part and the assembled ZIP before extracting.
3. Open SEMReview\\SEMReview.exe inside the extracted folder.
   Keep the _internal directory beside the executable. No Python install is needed.

CPU: works without NVIDIA hardware; training and inference run on the CPU.
NVIDIA CUDA 12.6: intended for supported GTX and RTX 20/30/40-series hardware.
RTX 50-series (Blackwell) is not supported by this CUDA build: use the CPU variant.
An NVIDIA driver compatible with CUDA 12.6 is required for GPU use.
The CUDA runtime is bundled; a separate CUDA Toolkit installation is not needed.

In training settings, enable the first download of official ResNet18 weights,
or select a previously downloaded official weights file for offline training.
No SEM images, project databases, or trained models are included in this download.

This executable is unsigned. Verify its source and the supplied SHA256 checksums.
Windows CI checks CPU training/inference and packaged startup. It has no GPU:
NVIDIA GPU performance, 8 GB VRAM usage, and real SEM accuracy are not validated.

The extraction helper uses a process-local PowerShell execution-policy override;
it does not change the computer's permanent execution-policy setting.
"""


def extraction_script(manifest_name: str) -> str:
    # manifest_name is constructed only from the fixed prefix, enum, and validated version.
    return r'''$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$temporary = $null
try {
    $manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot '__MANIFEST__') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.format_version -ne 1 -or $manifest.parts.Count -lt 2) { throw 'Invalid split ZIP manifest.' }
    if ([System.IO.Path]::GetFileName($manifest.archive) -ne $manifest.archive -or
        $manifest.archive -notmatch '^[A-Za-z0-9.-]+\.zip$') { throw 'Invalid archive filename.' }
    $archive = Join-Path $PSScriptRoot $manifest.archive
    $destination = Join-Path $PSScriptRoot ([System.IO.Path]::GetFileNameWithoutExtension($manifest.archive))
    if (Test-Path -LiteralPath $destination) { throw "Extraction folder already exists: $destination" }
    foreach ($part in $manifest.parts) {
        if ([System.IO.Path]::GetFileName($part.name) -ne $part.name) { throw 'Invalid part filename.' }
        $path = Join-Path $PSScriptRoot $part.name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing download: $($part.name)" }
        if ((Get-Item -LiteralPath $path).Length -ne $part.bytes) { throw "Incomplete download: $($part.name)" }
        Write-Host "Verifying $($part.name)..."
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $part.sha256) { throw "Checksum failed: $($part.name)" }
    }
    if (Test-Path -LiteralPath $archive) {
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $manifest.sha256) {
            throw 'An existing ZIP has different contents. Move it away before retrying.'
        }
    } else {
        $temporary = "$archive.assembling-$([guid]::NewGuid().ToString('N'))"
        $output = [System.IO.File]::Open($temporary, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
        try {
            foreach ($part in $manifest.parts) {
                $partStream = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $part.name))
                try { $partStream.CopyTo($output) } finally { $partStream.Dispose() }
            }
        } finally { $output.Dispose() }
        if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash -ne $manifest.sha256) {
            throw 'Assembled ZIP checksum failed.'
        }
        Move-Item -LiteralPath $temporary -Destination $archive
        $temporary = $null
    }
    Write-Host 'Extracting. This may take several minutes...'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($archive, $destination)
    Write-Host "Ready: $destination\SEMReview\SEMReview.exe"
    exit 0
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    if ($temporary -and (Test-Path -LiteralPath $temporary)) { Remove-Item -LiteralPath $temporary }
}
'''.replace("__MANIFEST__", manifest_name)


def package_windows(
    source: Path,
    output: Path,
    variant: str,
    version: str,
    *,
    max_part_bytes: int = DEFAULT_PART_BYTES,
    metadata: dict | None = None,
    readme: Path | None = None,
) -> list[Path]:
    """Package the app; each returned asset is smaller than GitHub's 2 GiB limit."""
    if variant not in VARIANTS:
        raise ValueError("Unknown Windows build variant")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-.][A-Za-z0-9.-]+)?", version):
        raise ValueError("Version must look like 0.1.0 or 0.1.0-rc.1")
    if max_part_bytes <= 0 or max_part_bytes >= 2 * 1024**3:
        raise ValueError("Part size must be positive and less than 2 GiB")
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Release output cannot be inside the application directory")
    if not (source / "SEMReview.exe").is_file():
        raise ValueError("Source must be a Windows PyInstaller folder containing SEMReview.exe")
    files = sorted(source.rglob("*"))
    if any(path.is_symlink() or path.is_junction() for path in files):
        raise ValueError("Release directory must not contain symbolic links")
    prefix = f"SEMReview-windows-x64-{variant}-v{version}"
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob(f"{prefix}*")) or list(output.glob(f"Extract-{prefix}*")):
        raise FileExistsError(f"Assets already exist for {prefix}")
    instructions = windows_instructions(variant)
    with tempfile.TemporaryDirectory(prefix=".sem-release-", dir=output) as staging_name:
        staging = Path(staging_name)
        archive = staging / f"{prefix}.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in files:
                if path.is_file():
                    bundle.write(path, (Path("SEMReview") / path.relative_to(source)).as_posix())
            bundle.writestr("SEMReview/WINDOWS-START.txt", instructions)
            if readme:
                bundle.write(readme, "SEMReview/README.md")
            bundle.writestr("SEMReview/BUILD-INFO.json", json.dumps({
                **(metadata or {}), "variant": variant, "release_version": version,
                "unsigned": True, "gpu_hardware_validated": False,
            }, indent=2, ensure_ascii=False))
        manifest = {
            "format_version": 1, "archive": archive.name, "bytes": archive.stat().st_size,
            "sha256": sha256(archive), "parts": [],
        }
        if archive.stat().st_size > max_part_bytes:
            with archive.open("rb") as stream:
                number = 1
                while stream.tell() < manifest["bytes"]:
                    part = staging / f"{archive.name}.{number:03d}"
                    remaining = max_part_bytes
                    with part.open("xb") as target:
                        while remaining:
                            block = stream.read(min(8 * 1024 * 1024, remaining))
                            if not block:
                                break
                            target.write(block)
                            remaining -= len(block)
                    manifest["parts"].append({"name": part.name, "bytes": part.stat().st_size, "sha256": sha256(part)})
                    number += 1
            archive.unlink()
            script_name = f"Extract-{prefix}.ps1"
            (staging / script_name).write_text(extraction_script(f"{prefix}.manifest.json"), encoding="utf-8-sig")
            (staging / f"Extract-{prefix}.cmd").write_text(
                '@echo off\r\n'
                f'powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0{script_name}"\r\n'
                'if errorlevel 1 (echo Extraction failed. See the message above.)\r\n'
                'pause\r\n', encoding="ascii", newline="",
            )
        else:
            manifest["parts"].append({"name": archive.name, "bytes": manifest["bytes"], "sha256": manifest["sha256"]})
        (staging / f"{prefix}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (staging / f"{prefix}.README.txt").write_text(instructions, encoding="utf-8")
        assets = sorted(staging.iterdir())
        checksums = staging / f"{prefix}.SHA256SUMS.txt"
        checksums.write_text("".join(f"{sha256(asset)}  {asset.name}\n" for asset in assets), encoding="ascii")
        assets.append(checksums)
        destinations = []
        for asset in assets:
            destination = output / asset.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(asset), destination)
            destinations.append(destination)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dist/SEMReview"))
    parser.add_argument("--output", type=Path, default=Path("release"))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--max-part-mib", type=int, default=1800)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig")) if args.metadata else {}
    assets = package_windows(args.source, args.output, args.variant, args.version,
                             max_part_bytes=args.max_part_mib * 1024**2, metadata=metadata, readme=args.readme)
    print(json.dumps([{"path": str(path), "bytes": path.stat().st_size} for path in assets], indent=2))


if __name__ == "__main__":
    main()
