from gamestate import GameState
import keyboard
import time
import os

gamestate = GameState()

while True:

    telemetry = gamestate.get_raw_telemetry()

    for key in telemetry:
        print(f"{key}: {telemetry[key]}")


    if keyboard.is_pressed("q"):
        print("Stopped.")
        break


    time.sleep(0.1)
    os.system("cls")

gamestate.close()