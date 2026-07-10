import logging
import os
import pathlib
import sys
import time
from pathlib import Path

from aidetector.utils.config import ConfigurationError

logger = logging.getLogger(__name__)
_RESTART_DELAY_SECONDS = 5


def _set_working_directory() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)


def _patch_windows_path_checkpoints() -> None:
    if os.name != "nt":
        setattr(pathlib, "WindowsPath", pathlib.PosixPath)


def start() -> None:
    _set_working_directory()
    _patch_windows_path_checkpoints()

    from aidetector.utils.config import load_config
    from aidetector.utils.onnx import setup_ort

    config = load_config()
    logger.info("Starting application with %d detector(s)", len(config.detectors))
    setup_ort(config)
    from aidetector.detection.manager import Manager

    manager = Manager.from_config(config)
    manager.start()
    try:
        manager.wait()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        manager.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    while True:
        try:
            start()
            return
        except ConfigurationError as error:
            logger.error("Application configuration is invalid: %s", error)
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
