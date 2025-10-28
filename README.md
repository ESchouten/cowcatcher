# AI Detector

Simple & lightweight detector that runs an Ultralytics YOLO model against video/image sources and exports detections to disk and/or Telegram.

## Usage

The `example/` folder contains:
- `config.json` - Sample configuration with detector settings, model URL, and Telegram options
- `compose.yml` - Docker Compose file using the pre-built image
- `sprong24.mp4` - Sample video file for testing

Run with Docker:

```bash
cd example
docker compose up -d
docker compose logs -f aidetector
docker compose down
```

## Development

Install dependencies and run locally:

```bash
uv sync
uv run main   # Expects a config.json in the root directory
```

Build and run docker image locally:

```bash
docker compose up --build
```

Development commands:

```bash
ruff check    # Lint
ruff format   # Format
ty check      # Type check
```

## Components

- Entrypoint: [`aidetector.main`](src/aidetector/__init__.py)
- Manager: loads detectors from config [`aidetector.manager.Manager`](src/aidetector/manager.py)
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

## Libraries

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO model inference
- [Pydantic](https://github.com/pydantic/pydantic) - Configuration deserialization

### Build tools

- [uv](https://github.com/astral-sh/uv) - Package manager
- [Ruff](https://github.com/astral-sh/ruff) - Linting and formatting
- [ty](https://github.com/hauntsaninja/ty) - Type checker

## License

This project is licensed under the AGPL (see [LICENSE](LICENSE)).