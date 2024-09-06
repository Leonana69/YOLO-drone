from typing import Union, Tuple, Optional
import numpy as np
import time, math
import cv2
from filterpy.kalman import KalmanFilter
from .shared_frame import SharedFrame

def iou(boxA, boxB):
    # Calculate the intersection over union (IoU) of two bounding boxes
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def euclidean_distance(boxA, boxB):
    centerA = ((boxA['x1'] + boxA['x2']) / 2, (boxA['y1'] + boxA['y2']) / 2)
    centerB = ((boxB['x1'] + boxB['x2']) / 2, (boxB['y1'] + boxB['y2']) / 2)
    return math.sqrt((centerA[0] - centerB[0])**2 + (centerA[1] - centerB[1])**2)


class ObjectInfo:
    def __init__(self, name, x, y, w, h) -> None:
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    def __str__(self) -> str:
        return f"{self.name} x:{self.x:.2f} y:{self.y:.2f} width:{self.w:.2f} height:{self.h:.2f}"

class ObjectTracker:
    def __init__(self, name, x, y, w, h) -> None:
        self.name = name
        self.kf_pos = self.init_filter()
        self.kf_siz = self.init_filter()
        self.timestamp = 0
        self.size = None
        self.update(x, y, w, h)

    def update(self, x, y, w, h):
        self.kf_pos.update((x, y))
        self.kf_siz.update((w, h))
        self.timestamp = time.time()

    def predict(self) -> Optional[ObjectInfo]:
        # if no update in 2 seconds, return None
        if time.time() - self.timestamp > 0.5:
            return None
        self.kf_pos.predict()
        self.kf_siz.predict()
        return ObjectInfo(self.name, self.kf_pos.x[0][0], self.kf_pos.x[1][0], self.kf_siz.x[0][0], self.kf_siz.x[1][0])

    def init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                        [0, 1, 0, 0]])
        kf.R *= 1  # Measurement uncertainty
        kf.P *= 1000  # Initial uncertainty
        kf.Q *= 0.01  # Process uncertainty
        return kf

class VisionSkillWrapper():
    def __init__(self, shared_frame: SharedFrame):
        self.shared_frame = shared_frame
        self.last_update = 0
        self.object_trackers: dict[str, ObjectTracker] = {}
        self.object_list = []
        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            cv2.aruco.DetectorParameters())
        
    def update(self):
        if self.shared_frame.timestamp == self.last_update:
            return
        self.last_update = self.shared_frame.timestamp
        self.object_list = []
        objs = self.shared_frame.get_yolo_result()['result']
        for obj in objs:
            name = obj['name']
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            self.object_list.append(ObjectInfo(name, x, y, w, h))
    def _update(self):
        if self.shared_frame.timestamp == self.last_update:
            return
        self.last_update = self.shared_frame.timestamp

        objs = self.shared_frame.get_yolo_result()['result']

        updated_trackers = {}

        for obj in objs:
            name = obj['name']
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']

            best_match_key = None
            best_match_distance = float('inf')
            
            # Find the best matching tracker
            for key, tracker in self.object_trackers.items():
                if tracker.name == name:
                    existing_box = {
                        'x1': tracker.kf_pos.x[0][0] - tracker.kf_siz.x[0][0] / 2,
                        'y1': tracker.kf_pos.x[1][0] - tracker.kf_siz.x[1][0] / 2,
                        'x2': tracker.kf_pos.x[0][0] + tracker.kf_siz.x[0][0] / 2,
                        'y2': tracker.kf_pos.x[1][0] + tracker.kf_siz.x[1][0] / 2,
                    }
                    distance = euclidean_distance(existing_box, box)
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_key = key

            # Update the best matching tracker or create a new one
            if best_match_key is not None and best_match_distance < 50:  # Threshold can be adjusted
                self.object_trackers[best_match_key].update(x, y, w, h)
                updated_trackers[best_match_key] = self.object_trackers[best_match_key]
            else:
                new_key = f"{name}_{len(self.object_trackers)}"  # Create a unique key
                updated_trackers[new_key] = ObjectTracker(name, x, y, w, h)

        # Replace the old trackers with the updated ones
        self.object_trackers = updated_trackers

        # Create the list of current objects
        self.object_list = []
        to_delete = []
        for key, tracker in self.object_trackers.items():
            obj = tracker.predict()
            if obj is not None:
                self.object_list.append(obj)
            else:
                to_delete.append(key)
        
        # Remove trackers that should be deleted
        for key in to_delete:
            del self.object_trackers[key]
    # def update(self):
    #     if self.shared_frame.timestamp == self.last_update:
    #         return
    #     self.last_update = self.shared_frame.timestamp
    #     objs = self.shared_frame.get_yolo_result()['result'] + self.shared_frame.get_yolo_result()['result_custom']
    #     for obj in objs:
    #         name = obj['name']
    #         box = obj['box']
    #         x = (box['x1'] + box['x2']) / 2
    #         y = (box['y1'] + box['y2']) / 2
    #         w = box['x2'] - box['x1']
    #         h = box['y2'] - box['y1']
    #         if name not in self.object_trackers:
    #             self.object_trackers[name] = ObjectTracker(name, x, y, w, h)
    #         else:
    #             self.object_trackers[name].update(x, y, w, h)
        
    #     self.object_list = []
    #     to_delete = []
    #     for name, tracker in self.object_trackers.items():
    #         obj = tracker.predict()
    #         if obj is not None:
    #             self.object_list.append(obj)
    #         else:
    #             to_delete.append(name)
    #     for name in to_delete:
    #         del self.object_trackers[name]

    def get_obj_list(self) -> str:
        self.update()
        str_list = []
        for obj in self.object_list:
            str_list.append(str(obj))
        return str(str_list).replace("'", '')

    def get_obj_info(self, object_name: str) -> ObjectInfo:
        for _ in range(10):
            self.update()
            for obj in self.object_list:
                if obj.name.startswith(object_name):
                    return obj
            time.sleep(0.2)
        return None

    def is_visible(self, object_name: str) -> Tuple[bool, bool]:
        return self.get_obj_info(object_name) is not None, False

    def object_x(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_x: {object_name} is not in sight', True
        return info.x, False
    
    def object_y(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_y: {object_name} is not in sight', True
        return info.y, False
    
    def object_width(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_width: {object_name} not in sight', True
        return info.w, False
    
    def object_height(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_height: {object_name} not in sight', True
        return info.h, False
    
    def object_distance(self, object_name: str) -> Tuple[Union[int, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_distance: {object_name} not in sight', True
        mid_point = (info.x, info.y)
        FOV_X = 0.42
        FOV_Y = 0.55
        if mid_point[0] < 0.5 - FOV_X / 2 or mid_point[0] > 0.5 + FOV_X / 2 \
        or mid_point[1] < 0.5 - FOV_Y / 2 or mid_point[1] > 0.5 + FOV_Y / 2:
            return 30, False
        depth = self.shared_frame.get_depth().data
        start_x = 0.5 - FOV_X / 2
        start_y = 0.5 - FOV_Y / 2
        index_x = (mid_point[0] - start_x) / FOV_X * (depth.shape[1] - 1)
        index_y = (mid_point[1] - start_y) / FOV_Y * (depth.shape[0] - 1)
        return int(depth[int(index_y), int(index_x)] / 10), False