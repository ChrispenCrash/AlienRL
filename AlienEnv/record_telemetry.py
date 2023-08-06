import time
import keyboard
import datetime
import os
import numpy as np
import pandas as pd
import torch
from telemetry import TelemetryData
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


timestamps = np.array([])

x_coords = np.array([])
y_coords = np.array([])
z_coords = np.array([])

gears = np.array([])
speeds = np.array([])
headings = np.array([])

normalizedCarPositions = np.array([])
wheels_off_track = np.array([])
off_tracks = np.array([])

fl_wheel_slip = np.array([])
fr_wheel_slip = np.array([])
rl_wheel_slip = np.array([])
rr_wheel_slip = np.array([])

fl_suspension_travel = np.array([])
fr_suspension_travel = np.array([])
rl_suspension_travel = np.array([])
rr_suspension_travel = np.array([])

force_feedback = np.array([])


info = TelemetryData()


print("Press 'q' to stop recording.")

print("Collecting telemetry...")

while True:

    timestamps = np.append(timestamps, datetime.datetime.now())

    x, z, y = list(info.graphics.carCoordinates)
    x_coords = np.append(x_coords, x)
    y_coords = np.append(y_coords, -1*y)
    z_coords = np.append(z_coords, z)

    # os.system('cls')
    # print("Car Coordinates")
    # print(f"x: {x:0.2f}\ty: {-1*y:0.2f}\tz: {z:0.2f}")
    

    gears = np.append(gears, info.physics.gear)
    speeds = np.append(speeds, info.physics.speedKmh)
    headings = np.append(headings, info.physics.heading)

    force_feedback = np.append(force_feedback, info.physics.finalFF)

    normalizedCarPositions = np.append(normalizedCarPositions, info.graphics.normalizedCarPosition)
    number_of_wheels_off_track = info.physics.numberOfTyresOut
    wheels_off_track = np.append(wheels_off_track, number_of_wheels_off_track)
    off_tracks = np.append(off_tracks, True if number_of_wheels_off_track >= 3 else False)

    fl, fr, rl, rr = list(info.physics.wheelSlip)
    fl_wheel_slip = np.append(fl_wheel_slip, fl)
    fr_wheel_slip = np.append(fr_wheel_slip, fr)
    rl_wheel_slip = np.append(rl_wheel_slip, rl)
    rr_wheel_slip = np.append(rr_wheel_slip, rr)

    fl_sus, fr_sus, rl_sus, rr_sus = list(info.physics.suspensionTravel)
    fl_suspension_travel = np.append(fl_suspension_travel, fl_sus)
    fr_suspension_travel = np.append(fr_suspension_travel, fr_sus)
    rl_suspension_travel = np.append(rl_suspension_travel, rl_sus)
    rr_suspension_travel = np.append(rr_suspension_travel, rr_sus)

    if keyboard.is_pressed('q'):
        print("...stopped.\n")
        break

    time.sleep(0.1)

info.close()

print("Saving telemetry...", end="")
telemetry_df = pd.DataFrame({
    "timestamps": timestamps,
    "x_coords": x_coords,
    "y_coords": y_coords,
    "z_coords": z_coords,
    "gears": gears,
    "speeds": speeds,
    "headings": headings,
    "normalizedCarPositions": normalizedCarPositions,
    "wheels_off_track": wheels_off_track,
    "off_track": off_tracks,
    "fl_wheel_slip": fl_wheel_slip,
    "fr_wheel_slip": fr_wheel_slip,
    "rl_wheel_slip": rl_wheel_slip,
    "rr_wheel_slip": rr_wheel_slip,
    "fl_suspension_travel": fl_suspension_travel,
    "fr_suspension_travel": fr_suspension_travel,
    "rl_suspension_travel": rl_suspension_travel,
    "rr_suspension_travel": rr_suspension_travel
})

telemetry_df.to_excel(f"data/wheelslip_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", index=False)
print("...saved.")