import time
from typing import Tuple
from .abs.robot_wrapper import RobotWrapper
from podtp import Podtp

class GearWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False
        config = {
            'ip': '192.168.0.115',
            'port': 80,
            'stream_port': 81
        }
        self.robot = Podtp(config)
        self.move_speed = 1
        self.rotate_speed = 15

    def keep_active(self):
        pass

    def connect(self):
        if not self.robot.connect():
            raise ValueError("Could not connect to the robot")
        if not self.robot.send_ctrl_lock(False):
            raise ValueError("Could not unlock the robot control")

    def takeoff(self) -> bool:
        return True

    def land(self):
        pass

    def start_stream(self):
        self.robot.start_stream()
        self.stream_on = True

    def stop_stream(self):
        self.robot.stop_stream()
        self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return self.robot.image_parser

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} cm")
        while distance > 0:
            self.robot.send_command_hover(0, self.move_speed, 0, 0)
            time.sleep(0.1)
            distance -= 10
        return True, False

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")
        return True, False

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")
        return True, False

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")
        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        if degree >= 90:
            print("-> Turning CCW over 90 degrees")
            return True, True
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        if degree >= 90:
            print("-> Turning CW over 90 degrees")
            return True, True
        return True, False
