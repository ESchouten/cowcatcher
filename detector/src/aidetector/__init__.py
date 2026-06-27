import logging
import os
import pathlib
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)
_RESTART_DELAY_SECONDS = 5


def _set_working_directory() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)


def _patch_windows_path_checkpoints() -> None:
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath


def start() -> None:
    _set_working_directory()
    _patch_windows_path_checkpoints()

    from aidetector.utils.config import config
    from aidetector.utils.onnx import setup_ort

    logger.info(f"Starting application with config: {config}")
    setup_ort(config)
    from aidetector.detection.manager import Manager

    manager = Manager.from_config(config)
    threads = manager.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        manager.stop()


def main():
    while True:
        try:
            start()
            return
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            return
        except Exception:
            logger.exception(
                "Application crashed, restarting in %ss", _RESTART_DELAY_SECONDS
            )
            time.sleep(_RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    main()
