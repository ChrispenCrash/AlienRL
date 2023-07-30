import time
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from AlienEnv.gamestate import GameState
from AlienEnv.controller import GameController

EPISODE_STEP_COUNT = 200

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

        self.step_count = 0

    def step(self, action):
        
        self.is_soft_reset = False
        self.is_hard_reset = False
        done = False
        truncated = False
        info = {}

        self.controller.set_inputs(action[0], action[1])

        # Retrieve the new state and telemetry
        framestack, telemetry, raw_telemetry = self.game_state.get_obs()

        temp_current_norm_position = raw_telemetry["normalizedCarPosition"]

        # Make sure game is not paused
        if raw_telemetry["paused"]:
            self.controller.pause_game()

        observation = {'framestack': framestack, 'telemetry': telemetry}

        reward, done = self._calculate_reward(raw_telemetry)

        # Check if the car is stuck
        current_coords = np.array([raw_telemetry["x"], raw_telemetry["y"], raw_telemetry["z"]])
        current_coords = np.round(current_coords, 2)
        if self.prev_coords is not None and np.array_equal(self.prev_coords, current_coords):
            # If the car hasn't moved for more than 10 seconds, reset the environment
            if time.time() - self.coords_updated_time > 10:
                done = True
                self.is_hard_reset = True
                reward -= 100
                print("Car stuck for 10 seconds, resetting.")
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

        return observation, reward, done, truncated, info
    
    def _calculate_reward(self, raw_telemetry):

        done = False

        speed = raw_telemetry['speed']
        current_norm_position = raw_telemetry['normalizedCarPosition']
        tyres_off_track = raw_telemetry['num_wheels_off_track']

        # Determine if the car is making progress
        if self.prev_norm_car_position is not None:
            progress = current_norm_position - self.prev_norm_car_position

            # Check if the car has crossed the finish line
            if progress < -0.5 and current_norm_position < 0.005:
                progress = (1 + current_norm_position) - self.prev_norm_car_position
        else:
            progress = 0

        # Max progress should be (0.005 * 1000) = 5
        progress_reward = progress * 50_000

        # Max speed ~285 km/h, so max reward is 285/50 = 5.7
        # speed_reward = max(speed,0) / 275
        speed_reward = 0 # temporarily disabled

        # Reward the car for making progress along the track
        progress_reward = max(0, progress_reward)  # Only reward for forward progress

        # Penalize the car if it goes off the track
        off_track_penalty = 0
        if tyres_off_track >= 3:
            off_track_penalty = -5
            # self.is_hard_reset = True

        # Penalize the car for going the wrong way
        if progress < -0.00001:
            progress_reward = -100
            self.is_hard_reset = True
            done = True

        # print(f"{progress_reward=}")
        # print(f"{speed_reward=}")
        # print(f"{off_track_penalty=}")
        # print(f"{current_norm_position=}")
        # print(f"{self.prev_norm_car_position=}")
        # print(f"{current_norm_position - self.prev_norm_car_position}")
        # print(speed_reward + progress_reward + off_track_penalty)
        # print()

        return (speed_reward + progress_reward + off_track_penalty), done

    def _calculate_episode_end(self, current_norm_car_position):
        multiple = math.ceil(current_norm_car_position/0.005)
        episode_end = round(multiple * 0.005, 3)
        return episode_end


    def reset(self, seed=None, rewind_time=20):

        self.step_count = 0

        super().reset(seed=seed)

        framestack, telemetry, raw_telemetry = self.game_state.get_obs()
        
        # If stuck in wall, rewind time 20 seconds
        self.rewind_time = rewind_time
        if self.is_soft_reset:
            self.controller.rewind_time(seconds=self.rewind_time)
            self.current_norm_car_position = raw_telemetry["normalizedCarPosition"]
            self.episode_end = self._calculate_episode_end(self.current_norm_car_position)
            self.episode_start = self.episode_end - 0.005
            self.is_soft_reset = False

        if self.is_hard_reset or raw_telemetry['num_wheels_off_track'] >= 3:
            self.controller.reset_inputs()
            self.controller.restart_session()
            self.current_norm_car_position = raw_telemetry["normalizedCarPosition"]
            self.episode_end = self._calculate_episode_end(self.current_norm_car_position)
            self.episode_start = self.episode_end - 0.005
            self.is_hard_reset = False

        self.prev_coords = None

        info = raw_telemetry
    
        return {'framestack': framestack, 'telemetry': telemetry}, info

    def close(self):
        self.controller.close()
        self.game_state.close()