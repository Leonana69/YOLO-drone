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