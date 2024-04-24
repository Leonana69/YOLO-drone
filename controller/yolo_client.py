from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np
from contextlib import asynccontextmanager

import json, os
import requests
import queue
import asyncio, aiohttp
import threading

from .utils import print_t
from .shared_frame import SharedFrame, Frame

DIR = os.path.dirname(os.path.abspath(__file__))

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
ROUTER_SERVICE_PORT = os.environ.get("ROUTER_SERVICE_PORT", "50049")

'''
Access the YOLO service through http.
'''
class YoloClient():
    def __init__(self, shared_frame: SharedFrame=None):
        self.service_url = 'http://{}:{}/yolo'.format(VISION_SERVICE_IP, ROUTER_SERVICE_PORT)
        self.image_size = (640, 352)
        self.frame_queue = queue.Queue() # queue element: (frame_id, frame)
        self.shared_frame = shared_frame
        self.frame_id = 0
        self.frame_id_lock = asyncio.Lock()

    def is_local_service(self):
        return VISION_SERVICE_IP == 'localhost'

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def plot_results(frame, results):
        if results is None:
            return
        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=50)
        w, h = frame.size
        for result in results:
            box = result["box"]
            draw.rectangle((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h), str_float_to_int(box["x2"], w), str_float_to_int(box["y2"], h)),
                        fill=None, outline='blue', width=4)
            draw.text((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 50), result["name"], fill='red', font=font)

    def plot_results_oi(frame, object_list):
        if object_list is None or len(object_list) == 0:
            return
        def str_float_to_int(value, multiplier):
            return int(float(value) * multiplier)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(os.path.join(DIR, "assets/Roboto-Medium.ttf"), size=50)
        w, h = frame.size
        for obj in object_list:
            draw.rectangle((str_float_to_int(obj.x - obj.w / 2, w), str_float_to_int(obj.y - obj.h / 2, h), str_float_to_int(obj.x + obj.w / 2, w), str_float_to_int(obj.y + obj.h / 2, h)),
                        fill=None, outline='blue', width=4)
            draw.text((str_float_to_int(obj.x - obj.w / 2, w), str_float_to_int(obj.y - obj.h / 2, h) - 50), obj.name, fill='red', font=font)

    def retrieve(self) -> Optional[SharedFrame]:
        return self.shared_frame
    
    @asynccontextmanager
    async def get_aiohttp_session_response(service_url, data, timeout_seconds=3):
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            # The ClientSession now uses the defined timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(service_url, data=data) as response:
                    response.raise_for_status()  # Optional: raises exception for 4XX/5XX responses
                    yield response
        except aiohttp.ServerTimeoutError:
            print_t(f"[Y] Timeout error when connecting to {service_url}")

    def detect_local(self, frame: Frame, conf=0.2):
        image = frame.image
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))
        self.frame_queue.put(frame)

        config = {
            'user_name': 'yolo',
            'stream_mode': True,
            'image_id': self.image_id,
            'conf': conf
        }
        files = {
            'image': ('image', image_bytes),
            'json_data': (None, json.dumps(config))
        }

        print_t(f"[Y] Sending request to {self.service_url}")

        response = requests.post(self.service_url, files=files)
        print_t(f"[Y] Response: {response.text}")
        json_results = json.loads(response.text)
        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get(), json_results)

    async def detect(self, frame: Frame, conf=0.2):
        if self.is_local_service():
            self.detect_local(frame, conf)
            return
        image = frame.image
        image_bytes = YoloClient.image_to_bytes(image.resize(self.image_size))

        async with self.frame_id_lock:
            self.frame_queue.put((self.frame_id, frame))
            config = {
                'user_name': 'yolo',
                'stream_mode': True,
                'image_id': self.image_id,
                'conf': conf
            }
            files = {
                'image': image_bytes,
                'json_data': json.dumps(config)
            }
            self.frame_id += 1

        async with YoloClient.get_aiohttp_session_response(self.service_url, files) as response:
            results = await response.text()

        try:
            json_results = json.loads(results)
        except:
            print_t(f"[Y] Invalid json results: {results}")
            return
        
        # discard old images
        if self.frame_queue.empty():
            return
        while self.frame_queue.queue[0][0] < json_results['image_id']:
            self.frame_queue.get()
        # discard old results
        if self.frame_queue.queue[0][0] > json_results['image_id']:
            return

        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get()[1], json_results)