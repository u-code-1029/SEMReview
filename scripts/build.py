"""Build a native application on the host OS: python scripts/build.py."""
from pathlib import Path
import os
import subprocess
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs

root = Path(__file__).resolve().parents[1]
args = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed",
    "--name", "SEMReview", "--paths", str(root / "src"),
    "--collect-data", "skimage", "--collect-data", "torchvision",
    "--collect-all", "imagecodecs", "--collect-submodules", "tifffile",
    "--hidden-import", "safetensors.torch", "--hidden-import", "semreview.worker",
    "--exclude-module", "PySide6.QtWebEngineCore", "--exclude-module", "PySide6.QtWebEngineWidgets",
    "--exclude-module", "PySide6.QtPdf", "--exclude-module", "matplotlib",
    "--exclude-module", "IPython", "--exclude-module", "tkinter",
    "--distpath", str(root / "dist"), "--workpath", str(root / "build"),
    "--specpath", str(root / "build"), str(root / "scripts/desktop_entry.py"),
]
# Newer torchvision wheels use _C_stable/image_stable and load them with
# torch.ops.load_library; older PyInstaller hooks only look for _C/image.
# Explicitly collect every native wheel library on both macOS and Windows.
native_libraries = collect_dynamic_libs("torchvision", search_patterns=["*.so", "*.dylib", "*.dll", "*.pyd"])
for source, target in native_libraries:
    args[-1:-1] = ["--add-binary", f"{source}{os.pathsep}{target}"]
env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(root / "build" / "pyinstaller-cache"))
subprocess.run(args, cwd=root, env=env, check=True)
print("Built:", root / "dist" / ("SEMReview.app" if sys.platform == "darwin" else "SEMReview"))
