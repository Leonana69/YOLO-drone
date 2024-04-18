import sys
from PIL import Image
sys.path.append("..")
from controller.yolo_grpc_client import YoloGRPCClient
from controller.shared_frame import Frame, SharedFrame

shared_frame = SharedFrame()
yolo_client = YoloGRPCClient(shared_frame=shared_frame)
frame = Frame(image=Image.open("./images/kitchen.webp"))
yolo_client.detect_local(frame)
print(shared_frame.get_yolo_result())
