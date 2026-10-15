"""PyInstaller entry point; supports both UI and --worker subprocess mode."""
from semreview.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
