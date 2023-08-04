import time
import os
import cv2
from PIL import Image
import numpy as np
import win32gui
import win32ui
import win32con
import torch
from AlienEnv.telemetry import TelemetryData
from AlienEnv.framestack import FrameStack
from collections import deque
from AlienEnv.utils import move_window_to_top_left
from torchvision import transforms
import threading
import pyautogui # Very important for window resize to work properly
import AlienEnv.config as cfg
from AlienEnv.utils import correct_heading

class BumperTransform:

    def __init__(self):
        self.resize = transforms.Resize((84,84))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __call__(self, frames):
        result = []
        for frame in frames:

            # Convert to PIL image
            pil_image = Image.fromarray(frame.astype('uint8'))

            # Crop
            width, height = pil_image.size
            pil_image = pil_image.crop((0, height // 2, width, height))

            # Resize
            resized_pil_image  = self.resize(pil_image)

            # Convert to tensor
            tensor = self.to_tensor(resized_pil_image )

            # Normalize
            # normalized_tensor = self.normalize(tensor)

            numpy_array = tensor.numpy()
            result.append(numpy_array)

            # result.append(normalized_tensor)

        return np.stack(result)
    
class BonnetTransform:

    def __init__(self):
        self.resize = transforms.Resize((84,84))
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.228, 0.228, 0.228])
        self.to_tensor = transforms.ToTensor()

    def __call__(self, frames):
        result = []
        for frame in frames:

            # Convert to PIL image
            pil_image = Image.fromarray(frame.astype('uint8'))
                        
            # print(f"Before: {pil_image.size}")

            # Crop
            width, height = pil_image.size
            pil_image = pil_image.crop((0, height // 2, width, 600))

            # Resize
            resized_pil_image = self.resize(pil_image)

            # Convert to tensor
            tensor = self.to_tensor(resized_pil_image)

            # Normalize
            # normalized_tensor  = self.normalize(tensor)

            numpy_array = tensor.numpy()
            result.append(numpy_array)

            # result.append(tensor)
            # result.append(normalized_tensor)

        return np.stack(result)

class GameState:
    def __init__(self):
    
        self.x = 0 # Start at x=0
        self.y = 45 # Start at y=45 for 4k monitor, 31 otherwise
        self.width = 1280
        self.height = 720
        # Not working for some reason
        # Fixed, pyautogui must be imported
        # move_window_to_top_left(x=-5, y=0, width=WIDTH, height=HEIGHT) 

        self.frame_stack = FrameStack(framestack_skip=4)
        self.frame_stack.start()

        self.telemetry = TelemetryData()

        # self.preprocess = BumperTransform()
        self.preprocess = BonnetTransform()

        self.last_packet_id = 0
        self.prev_rl_sus = None

        self.X_RANGE = cfg.MAX_X - cfg.MIN_X
        self.Y_RANGE = cfg.MAX_Y - cfg.MIN_Y
        self.Z_RANGE = cfg.MAX_Z - cfg.MIN_Z
    
    def get_cv2_frame(self):
        latest_frame = self.frame_stack.get_latest_frame()
        # https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
        return cv2.cvtColor(latest_frame, cv2.COLOR_BGRA2BGR)
    
    
    def get_framestack(self):
        frames = self.frame_stack.get_framestack()
        return self.preprocess(frames)

    def get_raw_telemetry(self):

        packetId = self.telemetry.graphics.packetId
        x, z, y = list(self.telemetry.graphics.carCoordinates)
        fl_ws, fr_ws, rl_ws, rr_ws = list(self.telemetry.physics.wheelSlip)
        fl_sus, fr_sus, rl_sus, rr_sus = list(self.telemetry.physics.suspensionTravel)
        front, rear, left, right, centre = list(self.telemetry.physics.carDamage)

        paused = False
        if (packetId == self.last_packet_id) or (self.prev_rl_sus == fl_sus):
            paused = True
        
        self.prev_rl_sus = fl_sus

        telemetry_dict = {}
        telemetry_dict["x"] = x
        telemetry_dict["y"] = -1*y
        telemetry_dict["z"] = z
        telemetry_dict["speed"] = self.telemetry.physics.speedKmh
        telemetry_dict["heading"] = correct_heading(self.telemetry.physics.heading)
        telemetry_dict["gear"] = self.telemetry.physics.gear
        telemetry_dict["steerAngle"] = self.telemetry.physics.steerAngle
        telemetry_dict["brake"] = self.telemetry.physics.brake
        telemetry_dict["gas"] = self.telemetry.physics.gas
        telemetry_dict["normalizedCarPosition"] = self.telemetry.graphics.normalizedCarPosition
        telemetry_dict["num_wheels_off_track"] = self.telemetry.physics.numberOfTyresOut
        telemetry_dict["car_damage"] = centre
        telemetry_dict["fl_ws"] = fl_ws
        telemetry_dict["fr_ws"] = fr_ws
        telemetry_dict["rl_ws"] = rl_ws
        telemetry_dict["rr_ws"] = rr_ws
        telemetry_dict["fl_sus"] = fl_sus
        telemetry_dict["fr_sus"] = fr_sus
        telemetry_dict["rl_sus"] = rl_sus
        telemetry_dict["rr_sus"] = rr_sus
        telemetry_dict["paused"] = paused

        return telemetry_dict
    
    def get_normalized_telemetry(self):
      
        # Normalize telemetry
        heading = self.telemetry.physics.heading
        steeringAngle = self.telemetry.physics.steerAngle
        gas = self.telemetry.physics.gas
        brake = self.telemetry.physics.brake
        gas_brake = gas - brake
        x, z, y = list(self.telemetry.graphics.carCoordinates)
        fl_ws, fr_ws, rl_ws, rr_ws = list(self.telemetry.physics.wheelSlip)
        fl_sus, fr_sus, rl_sus, rr_sus = list(self.telemetry.physics.suspensionTravel)

        one_hot_tyres = np.zeros(5)
        one_hot_tyres[self.telemetry.physics.numberOfTyresOut] = 1

        # Reducing telemetry
        # fl_ws_norm = np.clip((fl_ws - cfg.FL_WS_MEAN) / cfg.FL_WS_STD, -2, 2),
        # fr_ws_norm = np.clip((fr_ws - cfg.FR_WS_MEAN) / cfg.FR_WS_STD, -2, 2),
        # f_ws_norm = (fl_ws_norm[0] + fr_ws_norm[0])/2
        # rl_ws_norm = np.clip((rl_ws - cfg.RL_WS_MEAN) / cfg.RL_WS_STD, -2, 2),
        # rr_ws_norm = np.clip((rr_ws - cfg.RR_WS_MEAN) / cfg.RR_WS_STD, -2, 2),
        # r_ws_norm = (rl_ws_norm[0] + rr_ws_norm[0])/2
        
        norm_array = np.array([
            # (x - cfg.MIN_X ) / self.X_RANGE,
            # (y - cfg.MIN_Y ) / self.Y_RANGE,
            # (z - cfg.MIN_Z ) / self.Z_RANGE,
            max(self.telemetry.physics.speedKmh,0) / cfg.MAX_SPEED,
            steeringAngle,
            gas_brake,
            # np.sin(heading),
            # np.cos(heading),
            np.clip((fl_ws - cfg.FL_WS_MEAN) / cfg.FL_WS_STD, -2, 2),
            np.clip((fr_ws - cfg.FR_WS_MEAN) / cfg.FR_WS_STD, -2, 2),
            np.clip((rl_ws - cfg.RL_WS_MEAN) / cfg.RL_WS_STD, -2, 2),
            np.clip((rr_ws - cfg.RR_WS_MEAN) / cfg.RR_WS_STD, -2, 2),
            np.clip(fl_sus, 0.05, 0.13) / 0.13,
            np.clip(fr_sus, 0.05, 0.13) / 0.13,
            np.clip(rl_sus, 0.06, 0.14) / 0.14,
            np.clip(rr_sus, 0.06, 0.14) / 0.14
            # f_ws_norm,
            # r_ws_norm
        ])

        one_hot_tyres = np.zeros(5)
        one_hot_tyres[self.telemetry.physics.numberOfTyresOut] = 1

        norm_array = np.concatenate([norm_array, one_hot_tyres])

        # Change data type to float32
        norm_array = norm_array.astype(np.float32)

        return norm_array

    def get_obs(self):
        framestack = self.get_framestack()
        telemetry = self.get_normalized_telemetry()
        raw_telemetry = self.get_raw_telemetry()

        return framestack, telemetry, raw_telemetry
    
    def close(self):
        self.frame_stack.stop()
        self.telemetry.close()
        
    # def get_initial_observation(self):
    #     frames = self.get_framestack()
    #     telemetry = self.get_telemetry()

    #     return np.array(frames, telemetry)
