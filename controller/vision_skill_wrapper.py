from typing import Union, Tuple
from .shared_frame import SharedFrame

class VisionSkillWrapper():
    def __init__(self, shared_frame: SharedFrame):
        self.shared_frame = shared_frame

    def format_results(results):
        formatted_results = []
        for item in results['result']:
            box = item['box']
            name = item['name']
            x = round((box['x1'] + box['x2']) / 2, 2)
            y = round((box['y1'] + box['y2']) / 2, 2)
            w = round(box['x2'] - box['x1'], 2)
            h = round(box['y2'] - box['y1'], 2)
            info = f"{name} x:{x} y:{y} width:{w} height:{h}"
            formatted_results.append(info)
        return str(formatted_results).replace("'", '')

    def get_obj_list(self) -> str:
        return VisionSkillWrapper.format_results(self.shared_frame.get_yolo_result())

    def get_obj_info(self, object_name: str) -> dict:
        for item in self.shared_frame.get_yolo_result().get('result', []):
            # change this to start_with
            if item['name'].startswith(object_name):
                return item
        return None

    def is_visible(self, object_name: str) -> Tuple[bool, bool]:
        return self.get_obj_info(object_name) is not None, False
        
    def object_x(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_x: {object_name} is not in sight', True
        box = info['box']
        return (box['x1'] + box['x2']) / 2, False
    
    def object_y(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_y: {object_name} is not in sight', True
        box = info['box']
        return (box['y1'] + box['y2']) / 2, False
    
    def object_width(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_width: {object_name} not in sight', True
        box = info['box']
        return box['x2'] - box['x1'], False
    
    def object_height(self, object_name: str) -> Tuple[Union[float, str], bool]:
        info = self.get_obj_info(object_name)
        if info is None:
            return f'object_height: {object_name} not in sight', True
        box = info['box']
        return box['y2'] - box['y1'], False