# AI Detector

Lightweight detector that runs an Ultralytics YOLO model against video/image sources and exports detections to disk and/or Telegram.

## Instructions



## Build instructions

1. Provide a configuration file at [config.json](config.json) or use the example [example/config.json](example/config.json).
2. Run locally (after installing dependencies) with:
   - python -c "import aidetector; aidetector.main()"
   - or install the package and run the `main` entrypoint (see [pyproject.toml](pyproject.toml)).
3. Run in Docker using the provided [Dockerfile](Dockerfile) or compose via [compose.yml](compose.yml) / [example/compose.yml](example/compose.yml).

## Components

- Entrypoint: [`aidetector.main`](src/aidetector/__init__.py)
- Manager: loads detectors from config [`aidetector.manager.Manager.fromConfig`](src/aidetector/manager.py)
- Detector: runs the model and collects/export detections [`aidetector.detector.Detector`](src/aidetector/detector.py)
- Config types: [`aidetector.config.Config`](src/aidetector/config.py), collection & detector dataclasses
- Exporters:
  - Disk: [`aidetector.exporters.disk.DiskExporter`](src/aidetector/exporters/disk.py)
  - Telegram: [`aidetector.exporters.telegram.TelegramExporter`](src/aidetector/exporters/telegram.py)
  - Base type: [`aidetector.exporters.exporter.Exporter`](src/aidetector/exporters/exporter.py)

## Development

- Dependency and entrypoint configuration: [pyproject.toml](pyproject.toml)
- Multi-stage build Dockerfile [Dockerfile](Dockerfile)
- CI builds the container using [.github/workflows/ci.yaml](.github/workflows/ci.yaml)

## License

This project is licensed under the AGPL (see [LICENSE](LICENSE)).