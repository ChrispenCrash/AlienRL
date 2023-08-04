from AlienEnv.telemetry import TelemetryData
import os
import time
import keyboard
import math
import numpy as np
import pandas as pd
from math import radians
from AlienEnv.utils import correct_heading, get_nearest_points, get_line_direction_degrees, get_difference_in_degrees

telemetry = TelemetryData()

last_packet_id = 0
prev_rl_suspensionTravel = None
prev_norm_car_position = None

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

episode_length = 500
episode_number = 1
total_time_steps = 1
episode_time_steps = 1
episode_total_reward = 0
last_episode_reward = 0
print_episode = False

df = pd.read_parquet("AlienEnv/data/track_points.parquet")
df['order'] = df['order'].astype(int)
track_points = [tuple(x) for x in df.values.tolist()]

start_time = time.time()
last_episode_time = None

while True:

    current_fl_suspension = list(telemetry.physics.suspensionTravel)[0]

    if (telemetry.graphics.packetId == last_packet_id) or (prev_rl_suspensionTravel == list(telemetry.physics.suspensionTravel)[0]):
        print("Game is paused")
    else:

        current_norm_position = telemetry.graphics.normalizedCarPosition
        if prev_norm_car_position is None:
            prev_norm_car_position = current_norm_position

        tyres_off_track = telemetry.physics.numberOfTyresOut
        front, rear, left, right, centre = list(telemetry.physics.carDamage)
        car_damage = centre

        progress = current_norm_position - prev_norm_car_position

        progress_reward = progress * 10_000

        progress_reward_sig = np.tanh(progress_reward)


        if tyres_off_track == 0:
            on_track_reward = 1
        elif tyres_off_track == 1:
            on_track_reward = 0.75
        elif tyres_off_track == 2:
            on_track_reward = 0.5
        else:
            on_track_reward = -1

        if car_damage > 0:
            car_damage_reward = -1500
        else:
            car_damage_reward = 0

        distance_from_centre_line = 0

        ################################

        car_x, car_z, car_y = list(telemetry.graphics.carCoordinates)
        car_y = -1*car_y
        car_heading = correct_heading(telemetry.physics.heading)
        car_coords = (car_x, car_y)
        point1, point2 = get_nearest_points(track_points, car_coords)
        # track_direction = get_line_direction_degrees(point1, point2)
        theta = get_difference_in_degrees(car_heading, point1, point2)
        orientation_reward = np.round(np.cos(radians(theta)),2)

        ################################

        total_reward = progress_reward_sig + on_track_reward + car_damage_reward + distance_from_centre_line + orientation_reward

        episode_total_reward += total_reward

        if not print_episode:
            print(f"Episode: {episode_number}   Timesteps: {episode_time_steps}   Last episode reward: {last_episode_reward:.2f}")
            print("Progress | On Track |  Damage |  Angle |  Total")
            print(f"{progress_reward_sig:6.2f}   | {on_track_reward:6.1f}   | {car_damage_reward:7.1f} | {orientation_reward:6.2f} | {total_reward:6.1f}")
            print(f"\nActual progress: {progress_reward:6.2f}")
            if last_episode_time is not None:
                print(f"Last episode time: {last_episode_time:.2f} seconds")
        else:
            print(f"Episode: {episode_number} | Total Reward: {episode_total_reward}")

        prev_norm_car_position = current_norm_position
        total_time_steps += 1
        episode_time_steps += 1

    if keyboard.is_pressed("q"):
        print("Stopped.")
        break

    # average timestep length
    # time.sleep(0.042) # 0.1

    if (total_time_steps % episode_length == 0) or car_damage > 0:
        episode_time_steps = 1
        episode_number += 1
        finish_time = time.time()
        last_episode_time = finish_time - start_time
        start_time = finish_time
        last_episode_reward = episode_total_reward
        if print_episode:
            print(f"Episode: {episode_number} | Total Reward: {episode_total_reward}")
        if car_damage > 0:
            print("Car damage detected!")
            break
        episode_total_reward = 0

    if not print_episode:
        os.system("cls")

telemetry.close()
