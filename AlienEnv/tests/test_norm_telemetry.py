from gamestate import GameState
import keyboard
import time
import os

gamestate = GameState()

while True:

    telemetry = gamestate.get_normalized_telemetry()

    for val in telemetry:
        print(val)


    if keyboard.is_pressed("q"):
        print("Stopped.")
        break


    time.sleep(0.1)
    os.system("cls")

gamestate.close()