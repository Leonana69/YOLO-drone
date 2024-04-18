import asyncio
import threading
from PIL import Image
import time, sys
from io import BytesIO

sys.path.append("..")
from controller.yolo_grpc_client import YoloGRPCClient

import cv2
import grpc, json

asyncio_loop = asyncio.get_event_loop()
asyncio_thread = threading.Thread(target=asyncio_loop.run_forever)
asyncio_thread.start()

import proto.generated.hyrch_serving_pb2 as hyrch_serving_pb2
import proto.generated.hyrch_serving_pb2_grpc as hyrch_serving_pb2_grpc
channel = grpc.aio.insecure_channel(f'10.66.3.68:50050')
stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)

async def detect(image, conf=0.2):
    image_bytes = YoloGRPCClient.image_to_bytes(image.resize((640, 352)))

    detect_request = hyrch_serving_pb2.DetectRequest(image_id=12, image_data=image_bytes, conf=conf)
    response = await stub.DetectStream(detect_request)

    json_results = json.loads(response.json_data)
    print(json_results)

cap = cv2.VideoCapture(0)
def capture_loop(asyncio_loop):
    print("[C] Start capture loop...")
    count = 0
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        if ret:
            frame = Image.fromarray(frame)
            asyncio_loop.call_soon_threadsafe(asyncio.create_task, detect(frame))
            count += 1
        time.sleep(0.05)

capture_thread = threading.Thread(target=capture_loop, args=(asyncio_loop,))
capture_thread.start()
while True:
    time.sleep(1)
