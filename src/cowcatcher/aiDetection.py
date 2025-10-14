"""
Module beschrijving: wat doet deze module?
"""

from datetime import datetime

import cv2
import numpy
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
from collections import deque

#todo: remove line below
import config  # Import the config file
import telegramnotifier
import logging
import time
import os


from multithreading import MultiThreading
from config import Config  # Import je Config class
from ipcameras import IpCameras
from notifier import NotifierClass
from saveimage import SaveImage


class AiDetection(MultiThreading):
    """
    Interact with ip cameras grep a frame do the yolo magic and informs farmer if needed
    Runs periodicly based on a thres
    """

    # Class variabeles (gedeeld door alle instances)
    cap: cv2.VideoCapture

    def __init__(self, config: Config, notifier: NotifierClass, saveimage: SaveImage, ipcamera: IpCameras):
        
        self.logger = logging.getLogger(__name__)
        self.logger.debug("constructor, AiDetection called")

        #initialize parent class
        super().__init__(interval=config.get_detection_config()["frame_capture_interval_seconds"])
        self.logger.debug(f"thread interval: {config.get_detection_config()["frame_capture_interval_seconds"]}")
        
        #program control parameters
        retval : bool = False
        #todo: use correct type validating thingy self.last_detection_time: time = None

        # create local variables for the parameters passed to this constructor
        self.Config : Config = config
        self.notifier: NotifierClass = notifier
        self.ipcamera: IpCameras = ipcamera
        self.saveimage: SaveImage = saveimage
      
        # Constants for detection
        # lets store it in dict
        self.detection_settings = config.get_detection_config()

    
        # New: Deque for tracking confidence score progression
        self.confidence_history = deque(maxlen=10)  # Keep last 10 confidence scores (shorter for fast events)
        self.frame_history = deque(maxlen=10)  # Keep corresponding frames
        self.timestamp_history = deque(maxlen=10)  # Keep corresponding timestamps
    
        self.frame_count: int = 0 #frame counter whic are processed by the ai
        # Variables for collecting the best screenshots during an event
         
         # Startup init collection status
        self.last_detection_time = None  # Last time there was a detection above SAVE_THRESHOLD
        self.notification_counter = 0 
        self.stop_time = None


        self.collecting_screenshots = False
        self.peak_detected = False # Indicator if a peak was detected
        self.inactivity_period = 0  # Track how long there has been no activity
        self.collection_start_time = None
        self.event_detections: list = [] # List of tuples (confidence, frame, timestamp, original_image_path, results_obj)
        self.peak_detected = False # Indicator if a peak was detected
        self.inactivity_period = 0

        # Laad model
        self.logger.debug("Script started. Loading YOLO model...")
        # Load your trained YOLO model
        self.model = YOLO(ipcamera.ai_model)
        self.logger.debug("YOLO model successfully loaded")

        # todo: Error handling open camera
        retval = ipcamera.open_camera_stream()
        
        #todo 3: initialize the notifier => moet niet HIER! in main, maar 1 keer per start!!!
        # Send a start message to confirm everything is working
        # start_message = f"📋 Cowcatcher detection script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n⚠️ DISCLAIMER: Use at your own risk. This program uses Ultralytics YOLO and is subject to the GNU Affero General Public License v3.0 (AGPL-3.0)."
        # notifier.send_message(start_message)


    def stop_run(self):
        """ Stops de running of an thread!
            This does the cleanup of the objects!
        """        
        #Stop the threads
        self.logger.debug(f"{self.ipcamera.camera_name} - Calling super stop_run")
        super().stop_run()

        #stop de camera stream
        self.logger.debug(f"{self.ipcamera.camera_name} - Closing Camera stream")
        self.ipcamera.close_camera_stream()



    def get_stats(self) -> dict:
        """Get detection statistics with camera info.
            This function enriches the dictionary with camera specific data
            This is called when the thread is ended.
        """
        # Haal parent stats op
        stats = super().get_stats()
        
        # Voeg extra info toe, camera name
        stats['camera_name'] = self.ipcamera.camera_name
        
        return stats
    
    def execute(self):
        """
            Do the AI magic

            Handige short cuts:
            'camera_name' = self.ipcamera.camera_name


        """
        
        
        self.logger.debug(f"{self.ipcamera.camera_name} - Start Execute function")
        
        try:
            
            #################
            # INIT
            #################
            # read an frame from the IPcamera
            frame: numpy.ndarray = None # pyright: ignore[reportAssignmentType]
            frame = self.ipcamera.capture_camera_image()
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")

            #The framecounter
            self.frame_count += 1

            if self.frame_count % 100 == 0:
                self.logger.debug(f"{self.ipcamera.camera_name} - Frames processed: {self.frame_count}")



            # Only process every n frames
            if self.frame_count % self.detection_settings["process_every_n_frames"] == 0:
                
                self.logger.debug(f"{self.ipcamera.camera_name} - processed frame: {self.frame_count}") 
                
                # #################
                # Perform detection  
                # #################               
                results:list  = self.model.predict(source=frame, classes=[0], conf=0.2, verbose=False)
                # results:list  = self.model.predict(source="filename.mp4", classes=[0], conf=0.2, verbose=False)

                # 
                # #################
                # # Find highest confidence detection, in last captured frame
                # ################# 
                highest_conf_detection = None
                highest_conf = 0.0
                if len(results[0].boxes) > 0:
                    # Sort detections by confidence (highest first)
                    sorted_detections = sorted(results[0].boxes, key=lambda x: float(x.conf), reverse=True)
                    if len(sorted_detections) > 0:
                        highest_conf_detection = sorted_detections[0]
                        highest_conf = float(highest_conf_detection.conf)

                self.logger.debug(f"{self.ipcamera.camera_name} - highest_conf_detection: {highest_conf_detection}") 
                self.logger.debug(f"{self.ipcamera.camera_name} - highest_conf: {highest_conf}") 

                # Save detection in history (even at low confidence to see progression)
                if highest_conf_detection is not None:
                    self.confidence_history.append(highest_conf)
                    self.frame_history.append(frame.copy())  # Save a copy of the frame
                    self.timestamp_history.append(timestamp)
                else:
                    # Add zero if there's no detection (important for detecting progression)
                    self.confidence_history.append(0.0)
                    self.frame_history.append(frame.copy())
                    self.timestamp_history.append(timestamp)

                # 
                # #################
                # # Can Send Notificaton
                # ################# 
                can_send_notification = (self.last_detection_time is None or
                                            (current_time - self.last_detection_time).total_seconds() > self.detection_settings["cooldown_period"])

                # 
                # ###### FIRST FRAME DECTION ###########
                # # Is detection above save_threshold
                #
                # Start collecting screenshots if we have a first detection above SAVE_THRESHOLD
                # ################# 
                if highest_conf >= self.detection_settings["save_threshold"] and not self.collecting_screenshots and can_send_notification:
                    self.logger.debug(f"{self.ipcamera.camera_name} - Starting screenshot collection for {self.detection_settings["collection_time"]} seconds (searching for peak moment)")
                    self.collecting_screenshots = True
                    self.collection_start_time = current_time
                    self.event_detections = []
                    self.peak_detected = False
 
                    # Add any previous frames that are already in history (for context)
                    for i in range(len(self.confidence_history)):
                        if self.confidence_history[i] >= self.detection_settings["save_threshold"]:
                            # Get historical frame
                            hist_frame = self.frame_history[i]
                            hist_timestamp = self.timestamp_history[i]
                            hist_conf = self.confidence_history[i]

                            # Save the original frame
                            # todo: move to storage class
                            hist_original_save_path = os.path.join(self.detection_settings["save_folder"],
                                                                    f"mounting_detected{hist_timestamp}_conf{hist_conf:.2f}_history.jpg")
                            cv2.imwrite(hist_original_save_path, hist_frame)
                            #self.saveimage.write(hist_original_save_path,hist_frame)

                            # We don't have a result object for historical frames, so make a simple annotation
                            hist_annotated_frame = hist_frame.copy()
                            cv2.putText(hist_annotated_frame, f"Conf: {hist_conf:.2f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Add to event_detections (without result object, with annotated frame)
                            if self.detection_settings["show_life_feed"]:
                                self.event_detections.append(
                                    (hist_conf, hist_annotated_frame, hist_timestamp, hist_original_save_path, None))
                            else:
                                self.event_detections.append((hist_conf, None, hist_timestamp, hist_original_save_path, None))

                # If we are collecting
                if self.collecting_screenshots:
                    # If there's a detection, add to collection
                    if highest_conf_detection is not None and highest_conf >= self.detection_settings["save_threshold"]:
                        # Save original frame
                        original_save_path = os.path.join(self.detection_settings["save_folder"],
                                                            f"mounting_detected_{timestamp}_conf{highest_conf:.2f}.jpg")
                        cv2.imwrite(original_save_path, frame)
                        #self.saveimage.write(original_save_path,frame)

                        # Make annotated frame for live display, but don't save it yet
                        annotated_frame = results[0].plot()

                        # Add to collection - with annotated_frame for live display and results[0] for later annotations
                        if self.detection_settings["show_life_feed"]:
                            self.event_detections.append(
                                (highest_conf, annotated_frame.copy(), timestamp, original_save_path, results[0]))
                        else:
                            self.event_detections.append((highest_conf, None, timestamp, original_save_path, results[0]))

                        self.logger.debug(f"{self.ipcamera.camera_name} - Detection added to collection: {highest_conf:.2f}")

                        # Reset inactivity period when we have a detection above SAVE_THRESHOLD
                        self.inactivity_period = 0
                        self.last_detection_time = current_time

                        # Check if we have detected a peak
                        if highest_conf >= self.detection_settings["peak_detection_threshold"] and not self.peak_detected:
                            peak_detected = True
                            self.logger.debug(f"{self.ipcamera.camera_name} - Possible peak detected with confidence {highest_conf:.2f}")
                    else:
                        # If there's no detection above SAVE_THRESHOLD, increase inactivity period
                        if self.last_detection_time is not None:
                            inactivity_period = (current_time - self.last_detection_time).total_seconds()
                            if inactivity_period >= 2:  # Log every 2 seconds
                                self.logger.debug(f"{self.ipcamera.camera_name} - Inactivity period: {inactivity_period:.1f}s")

                    # Check if we should stop collecting
                    collection_duration = (current_time - self.collection_start_time).total_seconds() # pyright: ignore[reportOperatorIssue]

                    # Stop collecting if:
                    # 1. We had a peak moment and minimum collection time has passed, or
                    # 2. We reached maximum collection time, or
                    # 3. We have a very clear detection (above 0.85) - immediate response for clear cases, or
                    # 4. The inactivity period has reached self.detection_settings["inactivity_stop_time"] (no detections for X seconds)
                    if (self.peak_detected and collection_duration >= self.detection_settings["min_collection_time"]) or \
                            collection_duration >= self.detection_settings["collection_time"] or \
                            (highest_conf >= 0.85 and collection_duration >= 1) or \
                            self.inactivity_period >= self.detection_settings["inactivity_stop_time"]:
                        # Time to stop collecting and determine which images to send
                        self.logger.debug(
                            f"{self.ipcamera.camera_name} - Collection stopped after {collection_duration:.1f} seconds with {len(self.event_detections)} detections")

                        # Find the peak moment in the collected data
                        current_confidences = [conf for conf, _, _, _, _ in self.event_detections]

                        # NEW: Check if there are enough detections above NOTIFY_THRESHOLD
                        high_conf_detections = sum(1 for conf in current_confidences if conf >= self.detection_settings["notify_threshold"])
                        self.logger.debug(
                            f"{self.ipcamera.camera_name} - Number of high confidence detections: {high_conf_detections}/{self.detection_settings["min_high_confidence_detections"]} required")

                        # For short events: even with few detections, still send a notification
                        if len(current_confidences) > 0:
                            # Find the highest confidence and index
                            max_conf = max(current_confidences)
                            max_idx = current_confidences.index(max_conf)

                            # Determine which images we want to send
                            selected_indices = []

                            # For very short events with few frames
                            if len(current_confidences) <= 2:
                                # If we only have 1 or 2 frames, just send them all
                                selected_indices = list(range(len(current_confidences)))
                            else:
                                # For longer events: try to select before, peak, and after
                                # Add the peak
                                selected_indices.append(max_idx)

                                # Try to add an image before the peak - if it exists
                                if max_idx > 0:
                                    selected_indices.append(max(0, max_idx - 1))  # Directly before the peak

                                # Try to add an image after the peak - if it exists
                                if max_idx < len(self.event_detections) - 1:
                                    selected_indices.append(max_idx + 1)  # Directly after the peak

                            # Sort and limit to MAX_SCREENSHOTS
                            selected_indices = sorted(selected_indices)[:self.detection_settings["max_screenshots"]]

                            # MODIFIED: Send the selected images if we have both sufficient confidence
                            # and sufficient detections above the threshold
                            if max_conf >= self.detection_settings["notify_threshold"] and high_conf_detections >= self.detection_settings["min_high_confidence_detections"]:
                                # NEW: Increment the notification counter
                                self.notification_counter += 1

                                # NEW: Determine if sound should play (every 5th notification)
                                play_sound = (self.notification_counter % self.detection_settings["sound_every_n_notifications"] == 0)

                                for rank, idx in enumerate(selected_indices):
                                    conf, img, ts, original_path, results_obj = self.event_detections[idx]

                                    # For very short events (with 1-2 frames), use simpler labels
                                    if len(selected_indices) <= 2:
                                        stage = "Best capture" if idx == max_idx else "Extra capture"
                                    else:
                                        # Determine stage (before peak, peak, after peak)
                                        stage = "Before peak" if idx < max_idx else "Peak" if idx == max_idx else "After peak"

                                    # Determine which path to use based on configuration
                                    if self.detection_settings["send_annotated_images"]:
                                        # We need to save the annotated version for sending
                                        annotated_save_path = os.path.join(self.detection_settings["save_folder"],
                                                                            f"mounting_detected_{ts}_conf{conf:.2f}_annotated.jpg")

                                        # If the annotated image is already made (via live view), use it
                                        if img is not None:
                                            # Save the image that's already in memory
                                            #todo: replace with storage_class
                                            cv2.imwrite(annotated_save_path, img)
                                            #self.saveimage.write(annotated_save_path, img)
                                            send_path = annotated_save_path
                                        # Otherwise, if we have a result object, make a new annotation
                                        elif results_obj is not None:
                                            # Load the original frame
                                            # orig_frame = cv2.imread(original_path)
                                            orig_frame = self.saveimage.read(original_path)
                                            if orig_frame is not None:
                                                # Make a new annotation with the result object
                                                annotated_frame = results_obj.plot()
                                                # Save the annotated version
                                                cv2.imwrite(annotated_save_path, annotated_frame)
                                                #self.saveimage.write(annotated_save_path, annotated_frame)

                                                send_path = annotated_save_path
                                            else:
                                                # Fallback if the original frame cannot be loaded
                                                self.logger.debug(
                                                    f"{self.ipcamera.camera_name} - Could not load original frame for {original_path}, sending original")
                                                send_path = original_path
                                        else:
                                            # Fallback if we can't make an annotated version
                                            self.logger.debug(f"{self.ipcamera.camera_name} - No result object available for {ts}, sending original")
                                            send_path = original_path
                                    else:
                                        # Use the original path without annotations
                                        send_path = original_path

                                    # NEW: Message for Telegram with sound indicator
                                    sound_indicator = "🔊" if play_sound else "🔇"
                                    message = f"{sound_indicator} Mounting detected {self.ipcamera.camera_name} ({ts}) - Confidence: {conf:.2f}\n"
                                    message += f"Stage: {stage} - Rank {rank + 1}/{len(selected_indices)}\n"
                                    message += f"Event duration: {collection_duration:.1f}s\n"

                                    # NEW: Send to Telegram (sound off except for every 5th notification)
                                    # response = send_telegram_photo(send_path, message, disable_notification=not play_sound)
                                    response = self.notifier.send_photo(send_path, message, disable_notification=not play_sound)
                                    if response:
                                        sound_status = "WITH sound" if play_sound else "without sound"
                                        self.logger.debug(f"{self.ipcamera.camera_name} - Telegram message sent for {stage}: {conf:.2f} "
                                                f"{self.ipcamera.camera_name} - ({'with' if self.detection_settings["send_annotated_images"] and send_path != original_path else 'without'} bounding boxes) - {sound_status}")
                                    else:
                                        self.logger.error(f"{self.ipcamera.camera_name} - Telegram message sending failed for {stage}")

                                # Set last detection time for cooldown
                                self.last_detection_time = current_time
                                self.logger.debug(f"{self.ipcamera.camera_name} - Cooldown period of {self.detection_settings["cooldown_period"]} seconds started")
                                # NEW: Log sound status
                                if play_sound:
                                    self.logger.debug(f"{self.ipcamera.camera_name} - 🔊 SOUND NOTIFICATION #{self.notification_counter} sent!")
                                else:
                                    self.logger.debug(
                                        f"{self.ipcamera.camera_name} - 🔇 Silent notification #{self.notification_counter} sent (sound every {self.detection_settings["sound_every_n_notifications"]})")
                            else:
                                # MODIFIED: Give clear reason why no notification was sent
                                if max_conf < self.detection_settings["notify_threshold"]:
                                    self.logger.debug(
                                        f"{self.ipcamera.camera_name} - Highest confidence ({max_conf:.2f}) lower than NOTIFY_THRESHOLD ({self.detection_settings["notify_threshold"]}). No notification sent.")
                                elif high_conf_detections < self.detection_settings["min_high_confidence_detections"]:
                                    self.logger.debug(
                                        f"{self.ipcamera.camera_name} - Too few high confidence detections ({high_conf_detections}/{self.detection_settings["min_high_confidence_detections"]}). No notification sent.")

                        # Log reason for stopping collection
                        if self.inactivity_period >= self.detection_settings["inactivity_stop_time"]:
                            self.logger.debug(f"{self.ipcamera.camera_name} - Collection stopped due to inactivity ({self.inactivity_period:.1f}s without detections)")
                        elif highest_conf >= 0.85 and collection_duration >= 1:
                            self.logger.debug(f"{self.ipcamera.camera_name} - Collection stopped due to very high confidence detection ({highest_conf:.2f})")
                        elif self.peak_detected and collection_duration >= self.detection_settings["min_collection_time"]:
                            self.logger.debug(
                                f"{self.ipcamera.camera_name} - Collection stopped after peak detection and minimum collection time ({collection_duration:.1f}s)")
                        else:
                            self.logger.debug(f"{self.ipcamera.camera_name} - Collection stopped after maximum collection time ({collection_duration:.1f}s)")

                        # Reset collection status, 
                        # We are done detection, time to wait for new peak moment
                        self.collecting_screenshots = False
                        self.peak_detected = False
                        self.inactivity_period = 0
                
                #running headless now, so showing the life feed is not an option 
                # # Show the frame with detections only if self.detection_settings["show_life_feed"] is enabled
                # if self.detection_settings["show_life_feed"] and len(results) > 0:  # Check if there are results
                #     annotated_frame = results[0].plot()
                #     cv2.imshow("Cowcatcher Detection", annotated_frame)

            # # Key press to stop (q), but only check if the window is open
            # if self.detection_settings["show_life_feed"] and (cv2.waitKey(1) & 0xFF == ord('q')):
            #     self.logger.debug("User pressed 'q'. Script will stop.")
            #      break
        except Exception as e:
            self.logger.error(f"{self.ipcamera.camera_name} - Unexpected error: {str(e)}")
            stop_reason = f"{self.ipcamera.camera_name} - Stopped due to error: {str(e)}"
            self.notifier.send_message(stop_reason)






   
