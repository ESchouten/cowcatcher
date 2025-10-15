import json
from cowcatcher.config import Config
from cowcatcher.detector import Detector

config_json = json.load(open('test/config.json'))
config = Config(**config_json)

detectors = [Detector(detector) for detector in config.detectors]

def main():
    print(config)

if __name__ == "__main__":
    main()