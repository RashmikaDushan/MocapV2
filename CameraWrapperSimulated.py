import cv2 as cv
import time
import os
from dotenv import load_dotenv

class Camera:

    RES_SMALL = 0
    RES_LARGE = 1
    SUPPORT_CUSTOM_CONFIG = False
    _RESOLUTION = {RES_SMALL: (320, 240),
                   RES_LARGE: (640, 480)} # later use heigher resolutions

    def __init__(self, fps=90, resolution=1, gain=0, exposure=0):
        """
        Initialize the simulated camera with pre-defined image paths.

        :param fps: Frames per second (for simulation timing)
        :param resolution: Resolution (RES_SMALL or RES_LARGE)
        :param gain: Gain value (not functional in simulation)
        :param exposure: Exposure value (not functional in simulation)
        """

        load_dotenv()

        self.img_paths = [
            os.getenv("IMG_PATH_L"),
            os.getenv("IMG_PATH_R")
        ]

        self.num_cameras = 2
        self.working_camera_ids = [0, 1]

        for path in self.img_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image path {path} does not exist.")
            
        self.images = [cv.imread(img_path) for img_path in self.img_paths]

        # Resize images to match the resolution
        for i in range(len(self.images)):
            self.images[i] = cv.resize(self.images[i], (640, 480))

        self.fps = fps
        self.resolution = self._RESOLUTION[resolution]
        self.exposure_value = exposure # only for simulated camrea
        self.gain_value = gain # only for simulated camrea
        self.exposure = [exposure] * self.num_cameras  # Placeholder
        self.gain = [gain] * self.num_cameras  # Placeholder
        self.start_time = time.time()

    def release(self):
        """Release the simulated camera."""
        self.images = None

    def read(self, idx=None, timestamp=True, squeeze=True):
        """
        Simulate reading frames from cameras.

        :param idx: Camera index (0 or 1)
        :param timestamp: Include timestamps
        :param squeeze: Placeholder (not used)
        :return: Frames and timestamps
        """
        imgs = []
        ts = [1,1] # Placeholder
        del self.images
        self.images = [cv.imread(img_path) for img_path in self.img_paths]
        for i in range(len(self.images)):
            self.images[i] = cv.resize(self.images[i], self.resolution)
        imgs = self.images
        if timestamp:
            return imgs, ts
        else:
            return imgs

    def set_exposure(self, exposure): # Placeholder
        """Placeholder for setting exposure."""
        self.exposure_value = exposure

    def set_gain(self, gain): # Placeholder
        """Placeholder for setting gain."""
        self.gain_value = gain
