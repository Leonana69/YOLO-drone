from io import BytesIO
from PIL import Image
from typing import Optional, List

import json, sys, os
import queue
import grpc
import asyncio

from .yolo_client import SharedFrame, Frame
from .utils import print_t

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(PARENT_DIR, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
YOLO_SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050").split(",")[0]

'''
Access the YOLO service through gRPC.
'''
class YoloGRPCClient():
    def __init__(self, shared_frame: SharedFrame=None):
        channel = grpc.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        channel_async = grpc.aio.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)
        self.stub_async = hyrch_serving_pb2_grpc.YoloServiceStub(channel_async)
        self.image_size = (640, 352)
        self.frame_queue = queue.Queue()
        self.shared_frame = shared_frame
        self.frame_id_lock = asyncio.Lock()
        self.frame_id = 0

    def is_local_service(self):
        return VISION_SERVICE_IP == 'localhost'

    def image_to_bytes(image):
        # compress and convert the image to bytes
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()

    def retrieve(self) -> Optional[SharedFrame]:
        return self.shared_frame
    
    def set_class(self, class_names: List[str]):
        print_t(f"Set classes: {class_names}")
        class_request = hyrch_serving_pb2.SetClassRequest(class_names=class_names)
        self.stub.SetClasses(class_request)
    
    def detect_local(self, frame: Frame, conf=0.2):
        image = frame.image
        image_bytes = YoloGRPCClient.image_to_bytes(image.resize(self.image_size))
        self.frame_queue.put(frame)

        detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_bytes, conf=conf)
        response = self.stub.DetectStream(detect_request)
        
        json_results = json.loads(response.json_data)
        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get(), json_results)

    async def detect(self, frame: Frame, conf=0.1):
        if self.is_local_service():
            self.detect_local(frame, conf)
            return

        image = frame.image
        image_bytes = YoloGRPCClient.image_to_bytes(image.resize(self.image_size))
        async with self.frame_id_lock:
            image_id = self.frame_id
            self.frame_queue.put((self.frame_id, frame))
            self.frame_id += 1

        detect_request = hyrch_serving_pb2.DetectRequest(image_id=image_id, image_data=image_bytes, conf=conf)
        response = await self.stub_async.DetectStream(detect_request)
    
        json_results = json.loads(response.json_data)
        if self.frame_queue.empty():
            return
        # discard old images
        while self.frame_queue.queue[0][0] < json_results['image_id']:
            self.frame_queue.get()
        # discard old results
        if self.frame_queue.queue[0][0] > json_results['image_id']:
            return
        if self.shared_frame is not None:
            self.shared_frame.set(self.frame_queue.get()[1], json_results)