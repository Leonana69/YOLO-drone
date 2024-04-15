import sys
from PIL import Image
sys.path.append("..")
from controller.yolo_grpc_client import YoloGRPCClient

yolo_client = YoloGRPCClient()

image = Image.open("./images/kitchen.webp")
yolo_client.detect_local(image)
print(syr.get())