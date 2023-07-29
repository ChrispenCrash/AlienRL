import pyvjoy
from time import sleep
import pyautogui
import win32gui
import vgamepad as vg
import numpy as np

# XUSB_GAMEPAD_DPAD_UP
# XUSB_GAMEPAD_DPAD_DOWN
# XUSB_GAMEPAD_DPAD_LEFT
# XUSB_GAMEPAD_DPAD_RIGHT
# XUSB_GAMEPAD_START
# XUSB_GAMEPAD_BACK
# XUSB_GAMEPAD_LEFT_THUMB
# XUSB_GAMEPAD_RIGHT_THUMB
# XUSB_GAMEPAD_LEFT_SHOULDER
# XUSB_GAMEPAD_RIGHT_SHOULDER
# XUSB_GAMEPAD_GUIDE
# XUSB_GAMEPAD_A
# XUSB_GAMEPAD_B
# XUSB_GAMEPAD_X
# XUSB_GAMEPAD_Y

class GameController():
    def __init__(self, window_name="Assetto Corsa"):
        self.gamepad = vg.VX360Gamepad()

    def set_inputs(self, steering_angle=0, brake_pressure=0, throttle=0):
        # Set the steering angle, brake pressure, and gas
        self.gamepad.left_joystick_float(x_value_float=steering_angle, y_value_float=0.0)
        self.gamepad.left_trigger_float(value_float=brake_pressure)
        self.gamepad.right_trigger_float(value_float=throttle)
        self.gamepad.update()

    def reset_inputs(self):
        self.gamepad.reset()  # reset the gamepad to default state
        self.gamepad.update()

    def restart_session(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()
        sleep(3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        self.gamepad.update()

    def pause_game(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.gamepad.update()
        sleep(0.1)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.gamepad.update()

    def close(self):
        self.gamepad.reset()
        self.gamepad.update()
        del self.gamepad


if __name__ == "__main__":
    game_controller = GameController()
    game_controller.set_inputs(0, 0, 0)
    for i in np.arange(0,1.01,0.01):
        game_controller.set_inputs(0, i, i)
        sleep(0.05)

    sleep(2)
    print("Resetting inputs")
    game_controller.reset_inputs()

    sleep(2)
    print("Pausing game")
    game_controller.pause_game()
    sleep(1)
    print("Unpausing game")
    game_controller.pause_game()
    sleep(2)

    print("Resetting game")
    game_controller.restart_session()
    sleep(2)

    print("Accelerating right!!")
    game_controller.set_inputs(1,0,1)
    sleep(3)

    game_controller.close()