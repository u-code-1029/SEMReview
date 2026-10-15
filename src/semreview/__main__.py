"""Desktop and isolated worker entry point (also used in frozen builds)."""

from __future__ import annotations

import argparse
import multiprocessing
import sys

from . import __version__


def main() -> int:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="SEM 이미지 학습·검수")
    parser.add_argument("--project", help="프로젝트 폴더")
    parser.add_argument("--worker", metavar="COMMAND", help=argparse.SUPPRESS)
    parser.add_argument("--request", help=argparse.SUPPRESS)
    parser.add_argument("--cancel-file", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=f"SEM Review {__version__}")
    args = parser.parse_args()
    if args.worker:
        from .worker import run_worker

        return run_worker(args.worker, args.request, args.cancel_file)

    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("SEM Review")
    app.setOrganizationName("SEMReview")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    window = MainWindow(args.project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
