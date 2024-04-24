import time
from typing import Tuple
from .abs.robot_wrapper import RobotWrapper
from podtp import Podtp

DEFAULT_NO_VALID_READING = 100
SAFE_DISTANCE_THRESHOLD = 200
SIDE_DISTANCE_THRESHOLD = 70

def clean_sensor_data(raw_data):
    cleaned_data = raw_data[:]  # Create a copy of the raw data for cleaning

    for i in range(len(cleaned_data)):
        if cleaned_data[i] < 0:
            valid_previous = None
            valid_next = None

            # Find the previous valid value
            for j in range(i-1, -1, -1):
                if cleaned_data[j] >= 0:
                    valid_previous = cleaned_data[j]
                    break

            # Find the next valid value
            for k in range(i+1, len(cleaned_data)):
                if cleaned_data[k] >= 0:
                    valid_next = cleaned_data[k]
                    break

            # Decide what value to assign to the bad reading
            if valid_previous is not None and valid_next is not None:
                # Average if both previous and next valid values are found
                cleaned_data[i] = (valid_previous + valid_next) / 2
            elif valid_previous is not None:
                # Use the previous if only it is available
                cleaned_data[i] = valid_previous
            elif valid_next is not None:
                # Use the next if only it is available
                cleaned_data[i] = valid_next
            else:
                # If no valid readings are available, handle it with a default value or recheck
                cleaned_data[i] = DEFAULT_NO_VALID_READING

    return cleaned_data

def all_values_similar(distances, tolerance=100):
    # Check if all values are within a certain tolerance
    mean_distance = sum(distances) / len(distances)
    return all(abs(distance - mean_distance) < tolerance for distance in distances)

def significant_jump_detected(distances):
    # Detect significant jumps and evaluate sections
    jumps = []
    for i in range(1, len(distances)):
        if abs(distances[i] - distances[i - 1]) > 80 \
            and (distances[i] < SAFE_DISTANCE_THRESHOLD or distances[i - 1] < SAFE_DISTANCE_THRESHOLD):  # threshold for significant jump
            jumps.append(i)

    # Return list of jump positions
    return jumps

def evaluate_segments(distances, jumps, front=False):
    segments = []
    if len(jumps) == 0 and front:
        return segments
    start = 0
    for jump in jumps + [len(distances)]:
        segment = distances[start:jump]
        average_distance = sum(segment) / len(segment)
        if average_distance < SAFE_DISTANCE_THRESHOLD:  # Safe distance threshold
            segments.append({'start': start,
                             'end': jump,
                             'length': jump - start,
                             'average_distance': average_distance})
        start = jump
    return segments

def find_largest_gap(segments, distances):
    if not segments:
        return "forward"  # No free segments, continue forward

    # Find any segment longer than 4
    # freeway_segments = [seg for seg in segments if seg['length'] >= 3]
    # if freeway_segments:
    #     # Find the largest freeway segment
    #     largest_freeway = max(freeway_segments, key=lambda x: x['length'])
    #     mid_index = (largest_freeway['start'] + largest_freeway['end']) // 2
    #     if mid_index < len(distances) / 2:
    #         return "right"  # More space on the left, so turn right
    #     else:
    #         return "left"  # More space on the right, so turn left

    # # If no segment is longer than 4, choose the largest available segment
    # largest_segment = max(segments, key=lambda x: x['length'])
    # mid_index = (largest_segment['start'] + largest_segment['end']) // 2
    # if mid_index < len(distances) / 2:
    #     return "left"
    # else:
    #     return "right"
    weighted_sum = 0
    mass = 0
    for seg in segments:
        for i in range(seg['start'], seg['end']):
            weighted_sum += i
            mass += 1

    mid_index = weighted_sum // mass
    return "left" if mid_index > (len(distances) - 1) / 2 else "right"

class GearWrapper(RobotWrapper):
    def __init__(self):
        self.stream_on = False
        config = {
            'ip': '192.168.8.169',
            'ip2': '192.168.8.195',
            'port': 80,
            'stream_port': 81
        }
        self.robot = Podtp(config)
        self.move_speed_x = 2.5
        self.move_speed_y = 2.8
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
        return self.robot.sensor_data
    
    def front_blocked(self) -> int:
        front_dis = self.robot.sensor_data.depth.data[3,:]
        front_dis = clean_sensor_data(front_dis)
        # print(f"Front distance: {front_dis}")
        if min(front_dis) > SAFE_DISTANCE_THRESHOLD:
            return -1
        if all_values_similar(front_dis) and max(front_dis) < 150:
            return -2
        else:
            segments = evaluate_segments(front_dis, significant_jump_detected(front_dis), True)
            if len(segments) == 0:
                return -1
            direction = find_largest_gap(segments, front_dis)
            if direction == "forward":
                return -1
            elif direction == "left":
                return 1
            else:
                return 0
            
    def side_distance(self, left=True) -> int:
        index = 0 if left else 7
        left_dis = self.robot.sensor_data.depth.data[index,:]
        left_dis = clean_sensor_data(left_dis)
        segments = evaluate_segments(left_dis, significant_jump_detected(left_dis))
        min_dis = 99999
        for seg in segments:
            if seg['average_distance'] < min_dis:
                min_dis = seg['average_distance']
        return min_dis

    def move_forward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            vy = 0
            index = self.front_blocked()
            left_margin = self.side_distance(True)
            right_margin = self.side_distance(False)
            
            if left_margin > SIDE_DISTANCE_THRESHOLD and right_margin > SIDE_DISTANCE_THRESHOLD:
                vy = 0
            elif left_margin > SIDE_DISTANCE_THRESHOLD:
                vy = -self.move_speed_y
            elif right_margin > SIDE_DISTANCE_THRESHOLD:
                vy = self.move_speed_y
            else:
                if abs(left_margin - right_margin) > 50:
                    if left_margin < right_margin:
                        vy = self.move_speed_y
                    else:
                        vy = -self.move_speed_y

            if index == -2:
                if min(self.robot.sensor_data.depth.data[0,:]) > SAFE_DISTANCE_THRESHOLD:
                    self.turn_ccw(90)
                elif min(self.robot.sensor_data.depth.data[7,:]) > SAFE_DISTANCE_THRESHOLD:
                    self.turn_cw(90)
                else:
                    self.turn_ccw(180)
            if index != -1:
                if index == 0:
                    self.turn_cw(30)
                elif index == 1:
                    self.turn_ccw(30)
            self.robot.send_command_hover(0, self.move_speed_x, vy, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_backward(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, -self.move_speed_x, 0, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_left(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, 0, -self.move_speed_y, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_right(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} cm")
        self.robot.send_command_hover(0, 0, 0, 0)
        while distance > 0:
            self.robot.send_command_hover(0, 0, self.move_speed_y, 0)
            time.sleep(0.1)
            distance -= 2
        self.robot.send_command_hover(0, 0, 0, 0)
        return True, False

    def move_up(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} cm")
        return True, False

    def move_down(self, distance: int) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} cm")
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        self.robot.send_command_hover(0, 0, 0, 0)
        self.robot.send_command_position(0, 0, 0, degree)
        time.sleep(0.5 + degree / 70.0)
        self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CCW over 90 degrees")
        #     return True, True
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        self.robot.send_command_hover(0, 0, 0, 0)
        self.robot.send_command_position(0, 0, 0, -degree)
        time.sleep(0.5 + degree / 70.0)
        self.robot.send_command_hover(0, 0, 0, 0)
        # if degree >= 90:
        #     print("-> Turning CW over 90 degrees")
        #     return True, True
        return True, False
