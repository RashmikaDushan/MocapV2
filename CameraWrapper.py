import cv2 as cv
import time
import platform

class Camera:

    RES_SMALL = 0
    RES_LARGE = 1
    SUPPORT_CUSTOM_CONFIG = False
    _RESOLUTION = { RES_SMALL:(320,240),
                    RES_LARGE:(640,480) }
    
    def __init__(self,fps=90, resolution=1, gain=0, exposure=0):
        self.working_camera_ids = Camera.list_ports()
        self.cameras = []
        self.num_cameras = len(self.working_camera_ids)
        for camera_id in self.working_camera_ids:
            if platform.system() == 'Windows':
                camera = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
            else:
                camera = cv.VideoCapture(camera_id)
            if self.SUPPORT_CUSTOM_CONFIG:
                camera.set(cv.CAP_PROP_FRAME_WIDTH, self._RESOLUTION[resolution][0]) ## some cameras dont support custom resolutions,gain and exposure
                camera.set(cv.CAP_PROP_FRAME_HEIGHT, self._RESOLUTION[resolution][1])
                camera.set(cv.CAP_PROP_FPS, fps)
            self.cameras.append(camera)
        self.fps = fps
        self.resolution = self._RESOLUTION[resolution]
        self.exposure_value = exposure
        self.gain_value = gain
        self.exposure = [exposure]*self.num_cameras ## this is because this is how it was used in the original library
        self.gain = [gain]*self.num_cameras ## needs to be implemented properly

    def release(self):
        for camera in self.cameras:
            camera.release()

    def read(self, idx=None, timestamp=True, squeeze=True): ## squeeze needs to be figured out
        imgs = []
        ts = []
        for camera in self.cameras:
            ret, frame = camera.read()
            resized_frame = cv.resize(frame, self.resolution) ## resize to the resolution set
            # if not self.SUPPORT_CUSTOM_CONFIG:
                # resized_frame = cv.convertScaleAbs(resized_frame, alpha=self.gain_value, beta=self.exposure_value)
            ts.append(time.time())
            imgs.append(resized_frame)
        return imgs,ts
    
    def set_exposure(self, exposure): ## need to check if this works
        self.exposure_value = exposure
        if self.SUPPORT_CUSTOM_CONFIG:
            for camera in self.cameras:
                camera.set(cv.CAP_PROP_EXPOSURE, exposure)
    
    def set_gain(self, gain):
        self.gain_value = gain
        if self.SUPPORT_CUSTOM_CONFIG:
            for camera in self.cameras:
                camera.set(cv.CAP_PROP_GAIN, gain)
    
    @staticmethod
    def list_ports(): ## need to debug - the non existing ports give errors
        non_working_ports = [1,3,4,5,6,7,8,9]
        working_ports = []
        dev_port = 0
        max_ports_to_check = 10
        
        while dev_port < max_ports_to_check:
            if dev_port in non_working_ports:
                dev_port += 1
                continue
            try:
                if platform.system() == 'Windows':
                    camera = cv.VideoCapture(dev_port, cv.CAP_DSHOW)
                else:
                    camera = cv.VideoCapture(dev_port)
                if not camera.isOpened():
                    non_working_ports.append(dev_port)
                    print(f"Port {dev_port} is not working.")
                else:
                    is_reading, img = camera.read()
                    w = camera.get(cv.CAP_PROP_FRAME_WIDTH)
                    h = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
                    if is_reading:
                        print(f"Port {dev_port} is working and reads images ({w:.0f} x {h:.0f})")
                        working_ports.append(dev_port)
                    else:
                        print(f"Port {dev_port} is present but does not read images.")
            except Exception as e:
                pass
            finally:
                if 'camera' in locals():
                    camera.release()
            dev_port += 1

        return working_ports

