from PIL import Image
from typing import Optional
from numpy.typing import NDArray
import numpy as np
import threading
import time

class Frame():
    def __init__(self, image: Image.Image, depth: Optional[NDArray[np.int16]]=None):
        self._image = image
        self._depth = depth
    
    @property
    def image(self) -> Image.Image:
        return self._image
    
    @property
    def depth(self) -> Optional[NDArray[np.int16]]:
        return self._depth
    
    @image.setter
    def image(self, image: Image.Image):
        self._image = image

    @depth.setter
    def depth(self, depth: Optional[NDArray[np.int16]]):
        self._depth = depth

class SharedFrame():
    def __init__(self):
        self.timestamp = 0
        self.frame = Frame(None)
        self.yolo_result = {}
        self.lock = threading.Lock()

    def get_image(self) -> Optional[Image.Image]:
        with self.lock:
            return self.frame.image
    
    def get_yolo_result(self) -> dict:
        with self.lock:
            return self.yolo_result
    
    def get_depth(self) -> Optional[NDArray[np.int16]]:
        with self.lock:
            return self.frame.depth
        
    def set(self, frame: Frame, yolo_result: dict):
        with self.lock:
            self.frame = frame
            self.timestamp = time.time()
            self.yolo_result = yolo_result