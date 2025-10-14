
from datetime import datetime
import logging
import signal
import sys
import time
from saveimage import SaveImage

from config import Config  # Import the custom made config class to handle Config interactions
from ipcameras import IpCameras # Import the custom made IPCamera's class to handle stream 
from notifier import NotifierClass
from telegramnotifier import TelegramNotifierClass # Import the custom made notifier class to handle notifications to the farmer 
from homeassistantnotifier import HomeAssistantNotifierClass # Import the custom made notifier class to handle notifications to the farmer 
from aiDetection import AiDetection # Import the ai class where the magic happens


def initialize_logging ():
# Configureer logging 1x in main
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.getLogger('AiDetection').setLevel(logging.DEBUG)
    logging.getLogger('Config').setLevel(logging.DEBUG)
    logging.getLogger('IpCameras').setLevel(logging.DEBUG)
    logging.getLogger('TelegramNotifierClass').setLevel(logging.DEBUG)
    logging.getLogger('SaveImage').setLevel(logging.DEBUG)

def finalyze(detection: AiDetection,notifier: NotifierClass, reason : str):
        logger = logging.getLogger(__name__)
        logger.debug(f"{detection.ipcamera.camera_name} - Total frames processed: {detection.frame_count}")

        # Send a message that the script has stopped
        stop_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stop_reason = f"Reason:  {reason}"

        stop_message = f"{detection.ipcamera.camera_name} - ⚠️ WARNING: Cowcatcher detection script stopped at {stop_time}\n"
        stop_message += f"{stop_reason}\n"
        stop_message += f"Total frames processed: {detection.frame_count}\n"
        stop_message += f"Total notifications sent: {detection.notification_counter}"  # Show total notification count

        notifier.send_message(stop_message)
        logger.debug(f"{detection.ipcamera.camera_name} - Stop message sent to Telegram")

def main() -> int:
    """
    Main application logic.
    
    Returns:
        Exit code (0 = success, 1 = error)
    """

    detections = {}    #dict to save all detection instances

    initialize_logging()
    logger = logging.getLogger(__name__)

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        for camera_key, detection in detections.items():
            logger.debug(f"Stopping thread for {camera_key}...")
            detection.stop_run()
            finalyze(detection,notifier, "Script stopped (Keyboard interupt")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

  
    try:
        retval :bool = False
    
        logger.info("CowCatcherAI has started")

        # Maak config instance
        config = Config("config.json")

        # Folder for saving screenshots
        imagesaver = SaveImage(config.get_detection_config()["save_folder"])

        #determine right broker for dispatching messages
        broker = config.get_notifier_config("notifier")["broker"]
        if broker == "telegram":
            notifier = TelegramNotifierClass(config=config)
        elif broker == "homeassistant":
            notifier = HomeAssistantNotifierClass(config=config)
        else:
            raise ValueError(f"Unknown notifier type: {broker}")
        
        #test connection,by sending start message:
        start_message = f"📋 Cowcatcher detection script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n⚠️ DISCLAIMER: Use at your own risk. This program uses Ultralytics YOLO and is subject to the GNU Affero General Public License v3.0 (AGPL-3.0)."
        if not notifier.send_message(start_message):
            raise RuntimeError("Sending start notification failed, abort program")
         

        # Haal alle camera keys op
        camera_keys = config.get_all_ipcamera_keys()
        logger.debug(f"Found folowing cameras in config: {camera_keys}")
        # Output: ['ipcamera1', 'ipcamera2', 'ipcamera3', 'ipcamera4', 'ipcamera5']

        #Loop through the camera keys and create detection threads
        for camera_key in camera_keys:        
            # Maak de camera_instance
            camera = IpCameras(config, camera_key)
            #Maak de aiDetection instance
            detection = AiDetection(config, notifier, imagesaver, camera)
            #check if the camera is started up OKay
            if camera.get_status():
                # Sla op in dictionary met camera_key als key
                detections[camera_key] = detection
                #Start de detectie voor bovenstaande instance
                detection.start_run()


        # Keep main thread alive as log we have detections objects in our dict
        while len(detections) > 0 :
            # test camera detectie
            # detections["ipcamera1"].execute()
            time.sleep(1)
        
        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1

    

# MAIN
if __name__ == "__main__":
    initialize_logging()
    exit_code = main()
    exit(exit_code)

