from PIL import Image
import queue, time, os, json
from typing import Optional, Tuple
import asyncio
import uuid
from enum import Enum

from .shared_frame import SharedFrame, Frame
from .yolo_client import YoloClient
from .yolo_grpc_client import YoloGRPCClient
from .tello_wrapper import TelloWrapper
from .virtual_robot_wrapper import VirtualRobotWrapper
from .abs.robot_wrapper import RobotWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, HighLevelSkillItem, SkillArg
from .utils import print_t, input_t
from .minispec_interpreter import MiniSpecInterpreter, Statement


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMController():
    class RobotType(Enum):
        VIRTUAL = 0
        TELLO = 1
        GEAR = 2
    def __init__(self, robot_type, use_http=False, message_queue: Optional[queue.Queue]=None):
        self.shared_frame = SharedFrame()
        if use_http:
            self.yolo_client = YoloClient(shared_frame=self.shared_frame)
        else:
            self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
            self.yolo_client.set_class([])
        self.vision = VisionSkillWrapper(self.shared_frame)
        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            self.cache_folder = message_queue.get()

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        
        match robot_type:
            case LLMController.RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                self.drone: RobotWrapper = TelloWrapper()
            case LLMController.RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from .gear_wrapper import GearWrapper
                self.drone: RobotWrapper = GearWrapper()
            case _:
                print_t("[C] Start virtual drone...")
                self.drone: RobotWrapper = VirtualRobotWrapper()
        
        self.planner = LLMPlanner()

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        # self.low_level_skillset.add_skill(LowLevelSkillItem("move_in_circle", self.drone.move_in_circle, "Move in circle in cw/ccw", args=[SkillArg("cw", bool)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds", args=[SkillArg("seconds", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))

        self.low_level_skillset.add_skill(LowLevelSkillItem("goto", self.skill_goto, "goto the object", args=[SkillArg("object_name[*x-value]", str)]))
        # self.low_level_skillset.add_skill(LowLevelSkillItem("follow", self.skill_follow, "follow the object", args=[SkillArg("object_name", str), SkillArg("duration(s)", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))
        # load high-level skills
        self.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        with open(os.path.join(CURRENT_DIR, "assets/high_level_skills.json"), "r") as f:
            json_data = json.load(f)
            for skill in json_data:
                self.high_level_skillset.add_skill(HighLevelSkillItem.load_from_dict(skill))

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = self.high_level_skillset
        self.planner.init(high_level_skillset=self.high_level_skillset, low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()

    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_follow(self, object_name: str, duration: int) -> Tuple[None, bool]:
        print(f'Follow {object_name} for {duration} seconds')
        init_size = self.vision.object_width(object_name)[0]
        init_time = time.time()
        while time.time() - init_time < duration:
            if not self.vision.is_visible(object_name):
                print(f'Object {object_name} is not visible')
                break
            try:
                x = self.vision.object_x(object_name)[0]
                if x > 0.55:
                    self.drone.turn_cw(int((x - 0.5) * 70))
                elif x < 0.45:
                    self.drone.turn_ccw(int((0.5 - x) * 70))

                size = self.vision.object_width(object_name)[0]
                print(f'%%%%>>>> size: {init_size} {size}')
                if size < init_size * 0.8:
                    self.drone.move_forward(30)
                elif size > init_size * 1.2:
                    self.drone.move_backward(30)
                time.sleep(0.2)
            except Exception as e:
                print(f'Error: lost object {object_name}')
                break
        return None, False

    def skill_goto(self, object_name: str) -> Tuple[None, bool]:
        print(f'Goto {object_name}')
        if '[' in object_name:
            x = float(object_name.split('[')[1].split(']')[0])
        else:
            x = self.vision.object_x(object_name)[0]

        print(f'>> GOTO x {x} {type(x)}')

        if x > 0.55:
            self.drone.turn_cw(int((x - 0.5) * 70))
        elif x < 0.45:
            self.drone.turn_ccw(int((0.5 - x) * 70))

        self.drone.move_forward(110)
        return None, False

    def skill_take_picture(self) -> Tuple[None, bool]:
        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to {img_path}")
        self.append_message((img_path,))
        return None, False

    def skill_log(self, text: str) -> Tuple[None, bool]:
        self.append_message(f"[LOG] {text}")
        print_t(f"[LOG] {text}")
        return None, False
    
    def skill_re_plan(self) -> Tuple[None, bool]:
        return None, True

    def skill_delay(self, s: float) -> Tuple[None, bool]:
        time.sleep(s)
        return None, False

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def stop_controller(self):
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            # YoloClient.plot_results(image, self.shared_frame.get_yolo_result().get('result'))
            self.vision.update()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image
    
    def execute_minispec(self, minispec: str):
        interpreter = MiniSpecInterpreter(self.message_queue)
        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        return ret_val

    def execute_task_description(self, task_description: str):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        self.append_message('[TASK]: ' + task_description)
        ret_val = None
        while True:
            # set class for yolo
            # self.yolo_client.set_class(self.planner.get_class(task_description))
            self.current_plan = self.planner.plan(task_description, execution_history=self.execution_history)
            self.append_message(f'[Plan]: \\\\')
            # self.append_message(self.current_plan)
            # consent = input_t(f"[C] Get plan: {self.current_plan}, executing?")
            # if consent == 'n':
            #     print_t("[C] > Plan rejected <")
            #     return
            try:
                self.execution_time = time.time()
                ret_val = self.execute_minispec(self.current_plan)
            except Exception as e:
                print_t(f"[C] Error: {e}")
            break
            
            # disable replan for now
            if ret_val.replan:
                print_t(f"[C] > Replanning <: {ret_val.value}")
                continue
            else:
                break
        self.append_message(f'\n[Task ended]')
        # self.append_message(f'Task complete with {ret_val.value if ret_val else None}')
        self.append_message('end')
        self.current_plan = None
        self.execution_history = None

    def start_robot(self):
        print_t("[C] Connecting to robot...")
        self.drone.connect()
        print_t("[C] Starting robot...")
        self.drone.takeoff()
        self.drone.move_up(25)
        print_t("[C] Starting stream...")
        self.drone.start_stream()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        self.drone.stop_stream()
        self.controller_wait_takeoff = True

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        while self.controller_active:
            self.drone.keep_active()
            self.latest_frame = frame_reader.frame
            frame = Frame(frame_reader.frame,
                          frame_reader.depth if hasattr(frame_reader, 'depth') else None)

            if self.yolo_client.is_local_service():
                self.yolo_client.detect_local(frame)
            else:
                # asynchronously send image to yolo server
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
            time.sleep(0.10)
        # Cancel all running tasks (if any)
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        self.drone.stop_stream()
        self.drone.land()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")