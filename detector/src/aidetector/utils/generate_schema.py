import json
from pathlib import Path

from aidetector.utils.config import Config
from aidetector.exporters.metadata import DetectionMetadata
from pydantic import TypeAdapter


def main() -> None:
    config_directory = Path(__file__).resolve().parents[4] / "config"
    config = (
        TypeAdapter(Config).json_schema(),
        config_directory / "config.schema.json",
    )
    metadata = (
        TypeAdapter(DetectionMetadata).json_schema(),
        config_directory / "metadata.schema.json",
    )

    for schema, output_path in [config, metadata]:
        output_path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"Generated JSON schema: {output_path}")


if __name__ == "__main__":
    main()
