import json
import os
import shutil
import tempfile
from pathlib import Path

ORIGINAL_CWD = Path.cwd()
RUNTIME_DIR: Path | None = None


def pytest_configure():
    global RUNTIME_DIR

    RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="aidetector-tests-"))
    config = {"detectors": [{"detection": {"source": ["test-source"]}}]}
    (RUNTIME_DIR / "config.json").write_text(json.dumps(config), encoding="utf-8")
    os.chdir(RUNTIME_DIR)


def pytest_unconfigure():
    os.chdir(ORIGINAL_CWD)
    if RUNTIME_DIR is not None:
        shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
