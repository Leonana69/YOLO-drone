from PIL import Image
from typing import Optional
from numpy.typing import NDArray
import numpy as np
import threading
import time

class Frame():
    def __init__(self, image: Image.Image | NDArray[np.uint8]=None, depth: Optional[NDArray[np.int16]]=None):
        if image is None:
            self._image_buffer = np.zeros((352, 640, 3), dtype=np.uint8)
            self._image = Image.fromarray(self._image_buffer)
        if isinstance(image, np.ndarray):
            self._image_buffer = image
            self._image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            self._image = image
            self._image_buffer = np.array(image)
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
        self._image_buffer = np.array(image)

    @depth.setter
    def depth(self, depth: Optional[NDArray[np.int16]]):
        self._depth = depth

    @property
    def image_buffer(self) -> NDArray[np.uint8]:
        return self._image_buffer
    
    @image_buffer.setter
    def image_buffer(self, image_buffer: NDArray[np.uint8]):
        self._image_buffer = image_buffer
        self._image = Image.fromarray(image_buffer)

class SharedFrame():
    def __init__(self):
        self.timestamp = 0
        self.frame = Frame()
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