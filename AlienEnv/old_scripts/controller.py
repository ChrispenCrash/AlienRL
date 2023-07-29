import pyvjoy
import time
import pyautogui
import win32gui
import numpy as np

class GameController():
    def __init__(self, window_name="Assetto Corsa"):
        # Initialize the joystick
        self.joystick = pyvjoy.VJoyDevice(1)
        self.window_name = window_name
        if win32gui.FindWindow(None, self.window_name) == 0:
            raise Exception("Game window not found")

    def set_inputs(self, steering_angle, brake_pressure, gas):
        self.bring_to_front()
        # Set the steering angle, brake pressure, and gas
        self.joystick.set_axis(pyvjoy.HID_USAGE_X, int((steering_angle + 1) * 16383.5))
        self.joystick.set_axis(pyvjoy.HID_USAGE_Y, int(brake_pressure * 32767))
        self.joystick.set_axis(pyvjoy.HID_USAGE_Z, int(gas * 32767))

    def reset_inputs(self):
        self.bring_to_front()
        # Reset the steering angle, brake pressure, and gas
        self.joystick.set_axis(pyvjoy.HID_USAGE_X, int(16383.5))
        self.joystick.set_axis(pyvjoy.HID_USAGE_Y, 1)
        self.joystick.set_axis(pyvjoy.HID_USAGE_Z, 1)

    def reset_game(self):
        self.bring_to_front()
        # Press the key that resets the game
        with pyautogui.hold('ctrlleft'):
            pyautogui.press('r', interval=2)
            time.sleep(1)
            pyautogui.press('s')

    def pause_game(self):
        self.bring_to_front()
        # Press the key that pauses the game
        pyautogui.press('escape')


    def close(self):
        # Reset the joystick axis values to their default state
        self.joystick.set_axis(pyvjoy.HID_USAGE_X, int(16383.5))
        self.joystick.set_axis(pyvjoy.HID_USAGE_Y, 1)
        self.joystick.set_axis(pyvjoy.HID_USAGE_Z, 1)

    def windows_key(self):
        # Press the windows key
        pyautogui.press('win')

    def bring_to_front(self):
        # Bring the game window to the front
        window = win32gui.FindWindow(None, self.window_name)
        win32gui.SetForegroundWindow(window)


if __name__ == "__main__":
    game_controller = GameController()
    game_controller.set_inputs(0, 0, 0)
    for i in np.arange(0,1.01,0.01):
        game_controller.set_inputs(0, i, i)
        time.sleep(0.1)
    # game_controller.reset_game()

    # time.sleep(3)
    # game_controller.set_inputs(0, 0.5, 0.5)