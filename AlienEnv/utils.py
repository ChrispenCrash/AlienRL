import win32gui
import win32con
import pyautogui # Very important
from AlienEnv.config import GAME_WINDOW_NAME
import numpy as np
import pandas as pd
from math import pi, radians, degrees, atan2
from heapq import nsmallest
import math
from functools import lru_cache

def move_window_to_top_left(x=0, y=0, width=1280, height=720, window_name='Assetto Corsa'):
    # Find the window by its name
    window_handle = win32gui.FindWindow(None, window_name)

    left_padding = 11
    missing_width = 10 # No clue why but this is needed
    missing_height = 11 # No clue why but this is needed

    # If the window is found
    if window_handle:
        # Move and resize the window
        # Parameters: window handle, x, y, width, height
        win32gui.MoveWindow(window_handle, 0 - left_padding, 0, width + missing_width + 12, height + 45 + missing_height, True)
    else:
        print(f'Window - {window_name} - not found')

# @lru_cache()
def get_nearest_points(track_points, position, n=2):
    nearest_points = nsmallest(n, track_points, key=lambda track_point: math.hypot(position[0]-track_point[1], position[1]-track_point[2]))

    point1, point2 = nearest_points
    if point1[0] > point2[0]:
        next_point = point1
        prev_point = point2
    else:
        next_point = point2
        prev_point = point1

    return (prev_point[1], prev_point[2]), (next_point[1], next_point[2])

def get_line_direction_degrees(point1, point2):
    dx, dy = point2[0] - point1[0], point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))

def get_difference_in_degrees(car_heading_rad, point1, point2):
    # Convert car heading from radians to degrees
    car_heading_deg = math.degrees(car_heading_rad)

    # Get the direction of the road in degrees
    road_direction_deg = get_line_direction_degrees(point1, point2)

    # Calculate the difference in degrees
    difference = road_direction_deg - car_heading_deg

    # Normalize the difference to the range [-180, 180]
    difference = (difference + 180) % 360 - 180

    return difference

def correct_heading(heading):
    flipped_heading = heading * -1
    corrected_heading = flipped_heading - math.pi/2
    if corrected_heading < -math.pi:
        corrected_heading += 2*math.pi
    elif corrected_heading > math.pi:
        corrected_heading -= 2*math.pi
    return corrected_heading

def dist_to_line( point1, point2, car_coord):
    if (point2[0] - point1[0]) != 0:
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        A = m
        B = -1
        C = point1[1] - m*point1[0]
    else:
        A = 1
        B = 0
        C = -point1[0]

    dist = abs(A*car_coord[0] + B*car_coord[1] + C) / np.sqrt(A**2 + B**2)
    return dist

class ActionSmoother:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev_action = None

    def smooth(self, action):
        if self.prev_action is None:
            self.prev_action = action
            return action

        smoothed_action = self.alpha * self.prev_action + (1 - self.alpha) * action
        self.prev_action = smoothed_action
        return smoothed_action