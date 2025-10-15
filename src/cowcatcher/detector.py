from datetime import datetime
import os
from ultralytics import YOLO
from ultralytics.engine.results import Results

from cowcatcher.config import DetectorConfig

class Detector:
    def __init__(self, config: DetectorConfig):
        if config.save_directory is not None:
            os.makedirs(config.save_directory, exist_ok=True)
        self.config = config
        print(f"Loading model from {config.model_url}")
        self.model = YOLO(config.model_url)
        self.detect()

    def detect(self):
        print(self.config.save_directory)
        # ultralytics only supports one source for video files
        source = self.config.sources[0] if self.config.sources[0].name.endswith(('.mp4', '.avi', '.mov')) else self.config.sources
        results = self.model.predict(source, stream=True, conf=self.config.confidence_threshold) # pyright: ignore[reportUnknownMemberType]
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                self._save_image(result)
                    

    def _save_image(self, result: Results):
        if self.config.save_directory is not None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"detection_{now}.jpg"
            image_path = (os.path.join(self.config.save_directory, image_name))
            print(f"Detection made, saving to {image_path}")
            print(self.config.save_directory)
            print(image_name)
            result.save(image_path)