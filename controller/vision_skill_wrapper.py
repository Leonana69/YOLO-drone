from typing import Union, Tuple, Optional
import numpy as np
import time
import cv2
from filterpy.kalman import KalmanFilter
from .shared_frame import SharedFrame

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
        self.kf = self.init_filter()
        self.timestamp = 0
        self.size = None
        self.update(x, y, w, h)

    def update(self, x, y, w, h):
        self.kf.update((x, y))
        self.size = (w, h)
        self.timestamp = time.time()

    def predict(self) -> Optional[ObjectInfo]:
        # if no update in 2 seconds, return None
        if time.time() - self.timestamp > 0.8:
            return None
        self.kf.predict()
        return ObjectInfo(self.name, self.kf.x[0][0], self.kf.x[1][0], self.size[0], self.size[1])

    def init_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                        [0, 1, 0, 0]])
        kf.R *= 2  # Measurement uncertainty
        kf.P *= 1000  # Initial uncertainty
        kf.Q *= 0.01  # Process uncertainty
        return kf

class VisionSkillWrapper():
    def __init__(self, shared_frame: SharedFrame):
        self.shared_frame = shared_frame
        self.last_update = 0
        self.object_trackers = {}
        self.object_list = []
        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            cv2.aruco.DetectorParameters())
        
    def _update(self):
        if self.shared_frame.timestamp == self.last_update:
            return
        self.last_update = self.shared_frame.timestamp
        self.object_list = []
        objs = self.shared_frame.get_yolo_result()['result'] + self.shared_frame.get_yolo_result()['result_custom']
        for obj in objs:
            name = obj['name']
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            self.object_list.append(ObjectInfo(name, x, y, w, h))
    
    def update(self):
        if self.shared_frame.timestamp == self.last_update:
            return
        self.last_update = self.shared_frame.timestamp
        objs = self.shared_frame.get_yolo_result()['result'] + self.shared_frame.get_yolo_result()['result_custom']
        for obj in objs:
            name = obj['name']
            box = obj['box']
            x = (box['x1'] + box['x2']) / 2
            y = (box['y1'] + box['y2']) / 2
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']
            if name not in self.object_trackers:
                self.object_trackers[name] = ObjectTracker(name, x, y, w, h)
            else:
                self.object_trackers[name].update(x, y, w, h)

        locs, ids, _ = self.aruco_detector.detectMarkers(self.shared_frame.frame.image_buffer)
        for i, loc in enumerate(locs):
            x = (loc[0][0][0] + loc[0][1][0] + loc[0][2][0] + loc[0][3][0]) / 4 / self.shared_frame.frame.image.width
            y = (loc[0][0][1] + loc[0][1][1] + loc[0][2][1] + loc[0][3][1]) / 4 / self.shared_frame.frame.image.height + 0.1
            w = abs(loc[0][1][0] - loc[0][0][0]) / self.shared_frame.frame.image.width
            h = abs(loc[0][2][1] - loc[0][0][1]) / self.shared_frame.frame.image.height + 0.3
            name = f'door_{ids[i][0]}'
            if name not in self.object_trackers:
                self.object_trackers[name] = ObjectTracker(name, x, y, w, h)
            else:
                self.object_trackers[name].update(x, y, w, h)
        
        self.object_list = []
        to_delete = []
        for name, tracker in self.object_trackers.items():
            obj = tracker.predict()
            if obj is not None:
                self.object_list.append(obj)
            else:
                to_delete.append(name)
        for name in to_delete:
            del self.object_trackers[name]

    def get_obj_list(self) -> str:
        self.update()
        str_list = []
        for obj in self.object_list:
            str_list.append(str(obj))
        return str(str_list).replace("'", '')

    def get_obj_info(self, object_name: str) -> ObjectInfo:
        self.update()
        for obj in self.object_list:
            if obj.name.startswith(object_name):
                return obj
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