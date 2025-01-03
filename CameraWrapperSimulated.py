import cv2 as cv
import time
import os
import glob
import numpy as np

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


        self.images = []

        img_folder_paths = sorted([f"extrinsics/{i}" for i in os.listdir("extrinsics") if not i.startswith('.')])
        print(img_folder_paths)

        for img_folder_path in img_folder_paths:
            image_names = sorted(glob.glob(f'./{img_folder_path}/*.jpg'))
            print(image_names)
            imgs = []
            for fname in image_names:
                img = cv.imread(fname)
                imgs.append(img)
            self.images.append(imgs)

        self.images = np.array(self.images)
        self.images = self.images.swapaxes(0,1)
        print(self.images.shape)

        self.frame_number = 0

        self.num_cameras = 6

        # # Resize images to match the resolution
        # for i in range(len(self.images)):
        #     self.images[i] = cv.resize(self.images[i], (640, 480))

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
        imgs = self.images[self.frame_number]
        self.frame_number = (self.frame_number+1)%len(self.images)
        # for i in range(len(imgs)):
        #     imgs[i] = cv.resize(imgs[i], self.resolution)
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
