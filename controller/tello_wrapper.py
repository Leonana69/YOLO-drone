import time, cv2
import numpy as np
from typing import Tuple
from djitellopy import Tello

from .abs.robot_wrapper import RobotWrapper

import logging
Tello.LOGGER.setLevel(logging.WARNING)

MOVEMENT_MIN = 20
MOVEMENT_MAX = 300

SCENE_CHANGE_DISTANCE = 120
SCENE_CHANGE_ANGLE = 90

def adjust_exposure(img, alpha=1.0, beta=0):
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    # Apply exposure adjustment using the formula: new_img = img * alpha + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def sharpen_image(img):
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

class FrameReader:
    def __init__(self, fr):
        # Initialize the video capture
        self.fr = fr

    @property
    def frame(self):
        # Read a frame from the video capture
        frame = self.fr.frame
        frame = adjust_exposure(frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)
        
def cap_distance(distance):
    if distance < MOVEMENT_MIN:
        return MOVEMENT_MIN
    elif distance > MOVEMENT_MAX:
        return MOVEMENT_MAX
    return distance

class TelloWrapper(RobotWrapper):
    def __init__(self):
        self.drone = Tello()
        self.active_count = 0
        self.stream_on = False

    def keep_active(self):
        if self.active_count % 20 == 0:
            self.drone.send_control_command("command")
        self.active_count += 1

    def connect(self):
        self.drone.connect()

    def takeoff(self) -> bool:
        if not self.is_battery_good():
            return False
        else:
            self.drone.takeoff()
            return True

    def land(self):
        self.drone.land()

    def start_stream(self):
        self.stream_on = True
        self.drone.streamon()

    def stop_stream(self):
        self.stream_on = False
        self.drone.streamoff()

    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return FrameReader(self.drone.get_frame_read())

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_forward(cap_distance(distance))
        self.movement_x_accumulator += distance
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_back(cap_distance(distance))
        self.movement_x_accumulator -= distance
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_left(cap_distance(distance))
        self.movement_y_accumulator += distance
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_right(cap_distance(distance))
        self.movement_y_accumulator -= distance
        time.sleep(0.5)
        return True, distance > SCENE_CHANGE_DISTANCE

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_up(cap_distance(distance))
        time.sleep(0.5)
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        self.drone.move_down(cap_distance(distance))
        time.sleep(0.5)
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        self.drone.rotate_counter_clockwise(degree)
        self.rotation_accumulator += degree
        time.sleep(1)
        # return True, degree > SCENE_CHANGE_ANGLE
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        self.drone.rotate_clockwise(degree)
        self.rotation_accumulator -= degree
        time.sleep(1)
        # return True, degree > SCENE_CHANGE_ANGLE
        return True, False
    
    def is_battery_good(self) -> bool:
        self.battery = self.drone.query_battery()
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False