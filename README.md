# AI Detector

Simple & lightweight detector that runs an Ultralytics YOLO model against video/image sources and exports detections to disk and/or Telegram.

## Components

- Entrypoint: [`aidetector.main`](src/aidetector/__init__.py)
- Manager: creates and manages detectors from config [`aidetector.manager.Manager.fromConfig`](src/aidetector/manager.py)
- Detector: runs the model and collects/export detections [`aidetector.detector.Detector`](src/aidetector/detector.py)
- Config types: [`aidetector.config.Config`](src/aidetector/config.py)
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