"""
Module beschrijving: wat doet deze module?
"""
import cv2
import time
import numpy
import logging
from config import Config  # Import je Config class

class IpCameras:
    """
    Interact with ip cameras, or mp4 stream (test mode)
    """

    # Class variabeles (gedeeld door alle instances)
    cap: cv2.VideoCapture 

    def __init__(self, config: Config, json_top_level_key: str = "ipcamera1"):
    
        """
        Initialiseer een nieuwe instance van IpCamera.
        Or from an mp4 stream (experimenal)

        Args:
            config (instance of Config class): contains the representation of the config.json
            json_top_level_key (str, optional): contains the json_top_level_key from the camera to read from.
        """
        # Initialize the logger
        self.logger = logging.getLogger(__name__)
        
        # Get the config items from the config json under the toplevel key
        # which is provided as parameter to the constructor
        ipcameras_config = config.get_ipcamera_config(json_top_level_key)

        self.ip_address = ipcameras_config['ip_address'] # this can also be an pointer towards an mp4 file
        self.camera_name = ipcameras_config['camera_name']
        self.ai_model = ipcameras_config['ai_model']

        #internal parameter to keep track of health
        self._camera_health = True


    def open_camera_stream(self) -> bool:
        # Open the camera stream
        self.logger.info(f"{self.camera_name} - Opening camera stream...")
        self.cap = cv2.VideoCapture(self.ip_address)
        if not self.cap.isOpened():
            self.logger.error(f"{self.camera_name} - Cannot open camera\\mp4 stream on {self.ip_address}")
            self._camera_health = False
            return False
        else:
            self.logger.info(f"{self.camera_name} - Camera stream successfully opened")
            # TODO: what to do with life feed of camera_images?
            return True
            # self.logger.info(f"Live display is {'enabled' if SHOW_LIVE_FEED else 'disabled'}")

    def close_camera_stream(self) -> bool:
        #TODO: needs to be finished/
        # Close the camera stream
        self.logger.info(f"{self.camera_name} - Closing camera stream...")
        self.cap.release()
        return True

    def capture_camera_image(self) -> numpy.ndarray:
        """
        capture one fram from an IP camera stream

        Args:
            none, uses internal variables
        """
        #todo: improve error handling
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error(f"{self.camera_name} - Cannot read frame from camera")
            # Try to reconnect to the camera
            self.cap.release()
            time.sleep(5)
            self.open_camera_stream()
        return frame
    
    def get_status(self) -> bool:
        """
        Return the status of the internal health check

        Args:
            none, uses private internal variable
        """

        return self._camera_health

# Test de class:
if __name__ == "__main__":
    retval :bool = False
    # Maak config instance
    config = Config("config.json")

    camera_obj = IpCameras(config, "ipcamera_mp4_23")

    retval = camera_obj.open_camera_stream()
    frame = camera_obj.capture_camera_image()
    cv2.imwrite("test.jpeg", frame)
    

    retval = camera_obj.close_camera_stream()