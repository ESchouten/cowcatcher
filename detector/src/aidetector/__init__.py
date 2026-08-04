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


def _configuration_revision(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def start() -> bool:
    _set_working_directory()
    _patch_windows_path_checkpoints()

    from aidetector.utils.config import DEFAULT_CONFIG_PATH, load_config
    from aidetector.utils.onnx import setup_ort

    config_path = DEFAULT_CONFIG_PATH.resolve()
    config = load_config(config_path)
    loaded_revision = _configuration_revision(config_path)
    logger.info("Starting application with %d detector(s)", len(config.detectors))
    setup_ort(config)
    from aidetector.application import Application

    application = Application.from_config(config)
    application.start()
    try:
        reload_requested = application.wait(
            lambda: _configuration_revision(config_path) != loaded_revision
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        return False
    finally:
        application.stop()
    if reload_requested:
        logger.info("Configuration changed; restarting the detector")
    return reload_requested


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    while True:
        try:
            if not start():
                return
        except ConfigurationError as error:
            logger.error("Application configuration is invalid: %s", error)
            logger.info(
                "Waiting %ss for setup to create a valid config.json",
                _RESTART_DELAY_SECONDS,
            )
            try:
                time.sleep(_RESTART_DELAY_SECONDS)
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
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
