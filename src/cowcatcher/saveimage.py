import cv2
import os
import numpy as np
from pathlib import Path
from typing import Any

import logging

class SaveImage:
    def __init__(self, folder_name: str):
        """
        Initialize the ImageSaver with a folder name.
        
        Args:
            folder_name: Name of the folder where images will be saved
        """
        # Initialize the logger
        self.logger = logging.getLogger(__name__)


        self.folder_name = folder_name
        self.folder_path = None
        self.logger.info(f"Check and create: {self.folder_name}")
        #check if folder needs to be created
        self._check_and_create_folder()
    
    def _check_and_create_folder(self) -> bool:
        """
        Check if the folder exists, if not create it.
        
        Returns:
            bool: True if folder exists or was created successfully
        """
        try:
            self.folder_path = Path(self.folder_name)

            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
                self.logger.debug(f"Folder '{self.folder_path}' created")
            else:
                self.logger.warning(f"Folder '{self.folder_path}' already exists")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating folder: {e}")
            return False
    
    def read(self, filename: str) -> Any:
        """
        Read JPG image as frame.
        Args:
            filename: Name of the file (with .jpg extension)
        Returns:
            frame: numpy.ndarray containing the image data
        """
        # Read image
        orig_frame = cv2.imread(filename)
        
        # Check if image was loaded successfully
        if orig_frame is None:
            self.logger.error(f"Error: Could not read image from {filename}")
            return None 
        
        return orig_frame

    def write(self, filename: str, frame: np.ndarray) -> bool:
        """
        Save a frame as JPG image.
        
        Args:
            frame: numpy.ndarray containing the image data
            filename: Name of the file (with or without .jpg extension)
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        if self.folder_path is None:
            self._check_and_create_folder()
        
        # Ensure filename has .jpg extension
        if not filename.endswith('.jpg'):
            filename += '.jpg'
        
        file_path = "{self.folder_path}/{filename}"
        
        try:
            success = cv2.imwrite(str(file_path), frame)
            if success:
                self.logger.info (f"Image saved: {file_path}")
            return success
        except Exception as e:
            self.logger.info(f"Error saving image: {e}")
            return False


if __name__ == "__main__":
    # Create saver instance
    imagesaver = SaveImage("my_images")
    
    # Check/create folder
    # imagesaver.check_and_create_folder()
    
    # Create a dummy frame (or use your actual frame)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Save the frame
    imagesaver.write("test_image.jpg",dummy_frame)