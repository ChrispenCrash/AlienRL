from AlienEnv.telemetry import TelemetryData
import os
import time
import keyboard
from math import degrees

telemetry = TelemetryData()

last_packet_id = 0
prev_rl_suspensionTravel = None

while True:

    current_fl_suspension = list(telemetry.physics.suspensionTravel)[0]

    if (telemetry.graphics.packetId == last_packet_id) or (prev_rl_suspensionTravel == list(telemetry.physics.suspensionTravel)[0]):
        print("Game is paused")
    else:
        last_packet_id = telemetry.graphics.packetId
        rl_suspensionTravel = list(telemetry.physics.suspensionTravel)[0]
        print(telemetry.graphics.packetId)
        print(f"Car Model: {telemetry.static.carModel}")
        print(f"Track: {telemetry.static.track}")
        print(f"Position: {telemetry.graphics.position}")
        print()

        if telemetry.physics.numberOfTyresOut >= 3:
            print("Car is off the track!")

        print()
        print(f"Speed in km/h: {round(telemetry.physics.speedKmh,2)}")
        print(f"RPM: {round(telemetry.physics.rpms,2)}")
        print(f"Gear: {telemetry.physics.gear}")

        print()
        print(f"Steer_angle: {round(telemetry.physics.steerAngle,4)}")
        print(f"Throttle: {round(telemetry.physics.gas,4)}")
        print(f"Brake: {round(telemetry.physics.brake,4)}")

        # print()
        # print(f"Force Feedback: {telemetry.physics.finalFF}")

        print()
        print("Car Coordinates: ")
        x, z, y = list(telemetry.graphics.carCoordinates)
        print(f"x: {round(x,2)}")
        print(f"y: {round(-1*y,2)}")
        print(f"z: {round(z,2)}")

        print()
        print("Wheel Slip: ")
        fl, fr, rl, rr = list(telemetry.physics.wheelSlip)
        print(f"Front left: {round(rl,4)}")
        print(f"Front right: {round(fr,4)}")
        print(f"Rear left: {round(rl,4)}")
        print(f"Rear right: {round(rr,4)}")

        # Could use to detect if car hits wall
        print()
        front, rear, left, right, centre = list(telemetry.physics.carDamage)
        print(f"Car Damage: {centre}")

        print()
        print("Local Velocity: ")
        x_vel, z_vel, y_vel = telemetry.physics.localVelocity
        print(f"x: {round(x_vel,4)}")
        print(f"y: {round(y_vel,4)}")
        print(f"z: {round(z_vel,4)}")

        # print()
        # print("Local Angular Velocity: ")
        # print(f"x: {round(telemetry.physics.local_angular_vel.x,4)}")
        # print(f"y: {round(telemetry.physics.local_angular_vel.y,4)}")
        # print(f"z: {round(telemetry.physics.local_angular_vel.z,4)}")

        # print()
        # print("Tyre Contact Point: ")
        # fl, fr, rl, rr = list(telemetry.physics.tyreContactPoint)
        # print(f"Front left: {round(telemetry.physics.tyre_contact_point.front_left.x,4)}")
        # print(f"Front right: {round(telemetry.physics.tyre_contact_point.front_right.x,4)}")
        # print(f"Rear left: {round(telemetry.physics.tyre_contact_point.rear_left.x,4)}")
        # print(f"Rear right: {round(telemetry.physics.tyre_contact_point.rear_right.x,4)}")

        # print()
        # print("Tyre Contact Normal: ")
        # print(f"Front left: {round(telemetry.physics.tyre_contact_normal.front_left.x,4)}")
        # print(f"Front right: {round(telemetry.physics.tyre_contact_normal.front_right.x,4)}")
        # print(f"Rear left: {round(telemetry.physics.tyre_contact_normal.rear_left.x,4)}")
        # print(f"Rear right: {round(telemetry.physics.tyre_contact_normal.rear_right.x,4)}")

        # print()
        # print("Tyre Slip Ratio: ")
        # fl_slip, fr_slip, rl_slip, rr_slip = list(telemetry.physics.slipRatio)
        # print(f"Front left: {round(fl_slip,4)}")
        # print(f"Front right: {round(fr_slip,4)}")
        # print(f"Rear left: {round(rl_slip,4)}")
        # print(f"Rear right: {round(rr_slip,4)}")

        print()
        print(f"Heading in rad: {round(telemetry.physics.heading,4)}")
        print(f"Heading in degrees: {round(degrees(telemetry.physics.heading),2)}")
        print(f"Pitch: {round(telemetry.physics.pitch,4)}")
        print(f"Roll: {round(telemetry.physics.roll,4)}")

        print()
        print(f"Current Lap Time: {telemetry.graphics.currentTime}")
        print(f"Last Lap Time: {telemetry.graphics.lastTime}")
        print(
            f"Current Sector Index: {telemetry.graphics.currentSectorIndex+1}/{telemetry.static.sectorCount}"
        )
        print()
        print(f"Normalized Car Position: {telemetry.graphics.normalizedCarPosition}")
        # print(f"penalty: {telemetry.graphics.penalty}")
        # print(f"is_valid_lap: {telemetry.graphics.is_valid_lap}")
        print(f"Number of types out: {telemetry.physics.numberOfTyresOut}")

        # Doesn't work
        # print()
        # print(f"Player Car ID: {telemetry.graphics.player_car_id}")
        # for car_id in telemetry.graphics.car_id:
        #     print(f"Car ID: {car_id}")
        # print()
        # for car_coord in telemetry.graphics.car_coordinates:
        #     print(f"Car coord: {car_coord}")

        # else:
        #     print("Lost connection to ACC")

        print()
        fl, fr, rl, rr = list(telemetry.physics.tyreWear)
        print(f"Front left: {round(rl,4)}")
        print(f"Front right: {round(fr,4)}")
        print(f"Rear left: {round(rl,4)}")
        print(f"Rear right: {round(rr,4)}")

    if keyboard.is_pressed("q"):
        print("Stopped.")
        break

    # time.sleep(0.1)
    time.sleep(0.1)
    os.system("cls")

telemetry.close()
