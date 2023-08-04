import time
import math
import math
from math import radians
import numpy as np
import pandas as pd
import socket
import pickle
import gymnasium as gym
from gymnasium import spaces

from AlienEnv.gamestate import GameState
from AlienEnv.controller import GameController
from AlienEnv.utils import correct_heading, get_nearest_points, get_line_direction_degrees, get_difference_in_degrees

EPISODE_STEP_COUNT = 500

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def send_data(host, port, data):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, port))
        s.sendall(pickle.dumps(data))

df = pd.read_parquet("AlienEnv/data/track_points.parquet")
df['order'] = df['order'].astype(int)
track_points = [tuple(x) for x in df.values.tolist()]

class AlienRLEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': []}

    def __init__(self):

        super(AlienRLEnv, self).__init__()

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'framestack': spaces.Box(low=0, high=1, shape=(4,3,84,84), dtype=np.float32),
            'telemetry': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        })

        # Initialize the game controller and game state
        self.controller = GameController()
        self.game_state = GameState()

        self.rewind_time = 10

        # Store the previous coordinates and the time when they were updated
        self.prev_coords = None
        self.coords_updated_time = time.time()

        self.raw_telemetry = self.game_state.get_raw_telemetry()
        self.current_norm_car_position = self.raw_telemetry["normalizedCarPosition"]
        self.episode_end = self._calculate_episode_end(self.current_norm_car_position)
        self.episode_start = self.episode_end - 0.005

        self.prev_norm_car_position = self.current_norm_car_position

        self.is_hard_reset = False
        self.is_soft_reset = False

        self.prev_steer_action = None
        self.prev_throttle_action = None

        self.step_count = 0

    def step(self, action):
        
        self.is_soft_reset = False
        self.is_hard_reset = False
        done = False
        truncated = False
        info = {}

        self.curr_steer_action = action[0]
        self.curr_throttle_action = action[1]

        self.controller.set_inputs(action[0], action[1])

        # Retrieve the new state and telemetry
        framestack, telemetry, raw_telemetry = self.game_state.get_obs()

        temp_current_norm_position = raw_telemetry["normalizedCarPosition"]

        # Make sure game is not paused
        # if raw_telemetry["paused"]:
        #     self.controller.pause_game()

        observation = {'framestack': framestack, 'telemetry': telemetry}

        reward, done = self._calculate_reward(raw_telemetry)

        # # Check if the car is stuck
        current_coords = np.array([raw_telemetry["x"], raw_telemetry["y"], raw_telemetry["z"]])
        current_coords = np.round(current_coords, 0)
        if self.prev_coords is not None and np.array_equal(self.prev_coords, current_coords):
            # If the car hasn't moved for more than 10 seconds, reset the environment
            if time.time() - self.coords_updated_time > 20:
                done = True
                self.is_hard_reset = True
                reward = -1000
                print("Car stuck for 20 seconds, restarting.")
                self.coords_updated_time = time.time()
        else:
            # If the car has moved, update the coordinates and the time
            self.prev_coords = current_coords
            self.coords_updated_time = time.time()

        # Check if the car has reached the end of the episode
        # if raw_telemetry["normalizedCarPosition"] >= self.episode_end:
        #     done = True
        self.step_count += 1
        if self.step_count >= EPISODE_STEP_COUNT or self.is_hard_reset:
            done = True

        info = raw_telemetry

        self.prev_norm_car_position = temp_current_norm_position

        self.prev_steer_action = self.curr_steer_action
        self.prev_throttle_action = self.curr_throttle_action

        return observation, reward, done, truncated, info
    
    def _calculate_reward(self, raw_telemetry):

        done = False

        # speed = raw_telemetry['speed']
        current_norm_position = raw_telemetry['normalizedCarPosition']
        tyres_off_track = raw_telemetry['num_wheels_off_track']
        car_damage = raw_telemetry['car_damage']

        # Determine if the car is making progress
        if self.prev_norm_car_position is not None:
            progress = current_norm_position - self.prev_norm_car_position

            # Check if the car has crossed the finish line
            if progress < -0.5 and current_norm_position < 0.005:
                progress = (1 + current_norm_position) - self.prev_norm_car_position
        else:
            progress = 0

        # Penalize the car for going the wrong way
        if progress < -0.01:
            print("Car going the wrong way, resetting.")
            progress_reward = -1500
            self.is_hard_reset = True
            done = True
        else:
            progress = progress * 10_000
            # Max progress should be (0.005 * 1000) = 5
            progress_reward = tanh(progress)

        # Max speed ~285 km/h, so max reward is 285/50 = 5.7
        # speed_reward = max(speed,0) / 275
        # speed_reward = 0 # temporarily disabled

        # Reward the car for making progress along the track
        # progress_reward = max(0, progress_reward)  # Only reward for forward progress

        # Penalize large changes in steering angle and throttle
        # if self.prev_steer_action is not None:
        #     action_change = np.abs(self.curr_steer_action - self.prev_steer_action)
        #     action_change_penalty = - change_penalty_factor * action_change
        #     reward += action_change_penalty

        # Penalize the car if it goes off the track
        # off_track_penalty = 0
        # if tyres_off_track == 0:
        #     off_track_penalty= 0
        # elif tyres_off_track == 1:
        #     off_track_penalty = -0.2
        # elif tyres_off_track == 2:
        #     off_track_penalty = -0.3
        # else:
        #     off_track_penalty = -1
            # self.is_hard_reset = True

        # 0 tires = -0.0001
        # 1 tire  = -0.001
        # 2 tires = -0.01
        # 3 tires = -0.1
        # 4 tires = -1
        # off_track_penalty = -1*math.pow(0.1, 4-tyres_off_track)
        # if tyres_off_track == 0:
        #     off_track_penalty = 1
        if tyres_off_track == 0:
            on_track_reward = 1
        elif tyres_off_track == 1:
            on_track_reward = 0.75
        elif tyres_off_track == 2:
            on_track_reward = 0.5
        else:
            on_track_reward = -1

        if car_damage > 0:
            print("Car damaged, restarting.")
            car_damage_penalty = -1500
            self.is_hard_reset = True
            done = True
        else:
            car_damage_penalty = 0

        

        car_x = raw_telemetry['x']
        car_y = raw_telemetry['y']
        car_heading = raw_telemetry['heading']
        car_coords = (car_x, car_y)
        point1, point2 = get_nearest_points(track_points, car_coords)
        theta = get_difference_in_degrees(car_heading, point1, point2)
        orientation_reward = np.round(np.cos(radians(theta)),2)

        # print(f"{progress=}")
        # print(f"{progress_reward=}")
        # print(f"{car_damage_penalty=}")
        # print(f"{orientation_reward=}")
        # print(f"{on_track_reward=}")
        # print(f"{current_norm_position=}")
        # print(f"{self.prev_norm_car_position=}")
        # print(f"{current_norm_position - self.prev_norm_car_position}")
        # print(progress_reward + on_track_reward + car_damage_penalty + orientation_reward)
        # print()

        message = {"progress_reward": progress_reward, 
                   "on_track_reward": on_track_reward, 
                   "car_damage_penalty": car_damage_penalty, 
                   "orientation_reward": orientation_reward}
        send_data('localhost', 12345, message)

        return (progress_reward + on_track_reward + car_damage_penalty + orientation_reward), done

    def _calculate_episode_end(self, current_norm_car_position):
        multiple = math.ceil(current_norm_car_position/0.005)
        episode_end = round(multiple * 0.005, 3)
        return episode_end


    def reset(self, seed=None, rewind_time=20):

        self.step_count = 0

        super().reset(seed=seed)

        framestack, telemetry, raw_telemetry = self.game_state.get_obs()
        
        # If stuck in wall, rewind time 20 seconds
        # self.rewind_time = rewind_time
        # if self.is_soft_reset:
        #     self.controller.rewind_time(seconds=self.rewind_time)
        #     self.current_norm_car_position = raw_telemetry["normalizedCarPosition"]
        #     self.episode_end = self._calculate_episode_end(self.current_norm_car_position)
        #     self.episode_start = self.episode_end - 0.005
        #     self.is_soft_reset = False

        if self.is_hard_reset:
            self.is_hard_reset = False
            print("Hard reset detected, restarting session.")
            self.controller.reset_inputs()
            # breakpoint()
            self.controller.restart_session()
            self.coords_updated_time = time.time()
            # self.current_norm_car_position = raw_telemetry["normalizedCarPosition"]
            # self.episode_end = self._calculate_episode_end(self.current_norm_car_position)
            # self.episode_start = self.episode_end - 0.005
            # self.is_hard_reset = False

        self.prev_coords = None

        info = raw_telemetry
    
        return {'framestack': framestack, 'telemetry': telemetry}, info

    def close(self):
        self.controller.close()
        self.game_state.close()