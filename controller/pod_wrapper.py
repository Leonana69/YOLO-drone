import time, os
from typing import Tuple
from .abs.robot_wrapper import RobotWrapper
from podtp import Podtp

class PodWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False
        config = {
            'ip': '192.168.8.169',
            'port': 80,
            'stream_port': 81
        }
        self.robot = Podtp(config)
        self.height = 0
        self.unlock_count = 0
        self.move_speed = 0.2

    def keep_active(self):
        pass

    def connect(self):
        if not self.robot.connect():
            raise ValueError("Could not connect to the robot")

        if not self.robot.send_ctrl_lock(False):
            raise ValueError("Could not unlock the robot control")

    def takeoff(self) -> bool:
        self.robot.send_command_setpoint(0, 0, 0, 13000)
        time.sleep(1)
        self.height = 0.6
        self.robot.send_command_hover(self.height, 0, 0, 0)
        return True

    def land(self):
        for _ in range(0, 5):
            self.robot.send_command_hover(0.2, 0, 0, 0)
            time.sleep(0.2)
        self.robot.send_command_setpoint(0, 0, 0, 0)
        self.height = 0

    def start_stream(self):
        self.robot.start_stream()
        self.stream_on = True

    def stop_stream(self):
        self.robot.stop_stream()
        self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return self.robot.sensor_data
    
    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} cm")
        while distance > 0:
            self.robot.send_command_hover(self.height, -self.move_speed, 0, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(self.height, 0, 0, 0)
        return True, False

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")
        while distance > 0:
            self.robot.send_command_hover(self.height, self.move_speed, 0, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(self.height, 0, 0, 0)
        return True, False

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")
        # self.robot.send_command_hover(0, 0, 0, 0)
        # while distance > 0:
        #     self.robot.send_command_hover(0, 0, -self.move_speed_y, 0)
        #     time.sleep(0.1)
        #     distance -= 2
        # self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")
        # self.robot.send_command_hover(0, 0, 0, 0)
        # while distance > 0:
        #     self.robot.send_command_hover(0, 0, self.move_speed_y, 0)
        #     time.sleep(0.1)
        #     distance -= 2
        # self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        # self.height += distance / 100
        # self.robot.send_command_hover(self.height, 0, 0, 0)
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        # self.height -= distance / 100
        # self.robot.send_command_hover(self.height, 0, 0, 0)
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        # self.robot.send_command_hover(0, 0, 0, 0)
        # self.robot.send_command_position(0, 0, 0, degree)
        # time.sleep(1 + degree / 50.0)
        # self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CCW over 90 degrees")
        #     return True, True
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        # self.robot.send_command_hover(0, 0, 0, 0)
        # self.robot.send_command_position(0, 0, 0, -degree)
        # time.sleep(1 + degree / 50.0)
        # self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CW over 90 degrees")
        #     return True, True
        return True, False
    
    def move_in_circle(self, cw) -> Tuple[bool, bool]:
        # if cw:
        #     vy = -8
        #     vr = -12
        # else:
        #     vy = 8
        #     vr = 12
        # for i in range(50):
        #     self.robot.send_command_hover(0, 0, vy, vr)
        #     time.sleep(0.1)
        # self.robot.send_command_hover(0, 0, 0, 0)
        return True, False