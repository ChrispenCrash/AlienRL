import pyvjoy
from time import sleep, time
import keyboard
import pyautogui
import win32gui, win32con
import vgamepad as vg
import numpy as np
from pynput.keyboard import Controller as keyboardController
import ctypes

pyautogui.FAILSAFE = False

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
KEYEVENTF_KEYDOWN = 0x0
KEYEVENTF_KEYUP = 0x2

class GameController():
    def __init__(self, window_name="Assetto Corsa"):
        self.gamepad = vg.VX360Gamepad()

    def set_inputs(self, steering_angle=0, brake_throttle_pressure=0):

        throttle_pressure = max(0, brake_throttle_pressure)
        brake_pressure = min(0, brake_throttle_pressure) * -1

        # Set the steering angle, brake pressure, and gas
        self.gamepad.left_joystick_float(x_value_float=steering_angle, y_value_float=0.0)
        self.gamepad.left_trigger_float(value_float=brake_pressure)
        self.gamepad.right_trigger_float(value_float=throttle_pressure)
        self.gamepad.update()

    def reset_inputs(self):
        self.gamepad.reset()  # reset the gamepad to default state
        self.gamepad.update()

    def restart_session(self):
        # self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        # self.gamepad.update()
        # sleep(3)
        # self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        # self.gamepad.update()

        # self.focus_window()
        # ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYDOWN, 0)
        # ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYUP, 0)
        self.reset_inputs()
        pyautogui.click(x=250, y=25)
        sleep(0.5)
        ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYDOWN, 0)
        sleep(0.25)
        ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYUP, 0)
        sleep(1)
        pyautogui.click(x=645, y=426)
        sleep(0.5)
        pyautogui.click(x=645, y=426)
        sleep(1)
        pyautogui.click(x=49, y=217)
        sleep(1)
        # pyautogui.click(x=49, y=217)
        # sleep(0.1)
        # click off the window
        pyautogui.click(x=647, y=881)

    def pause_game(self):
        # self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        # self.gamepad.update()
        # sleep(0.1)
        # self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        # self.gamepad.update()
        self.focus_window()
        sleep(0.25)
        ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYDOWN, 0)
        sleep(0.5)
        ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYUP, 0)
        pyautogui.click(x=647, y=881)

    def focus_window(self):
        # Bring window to front
        # window_handle = win32gui.FindWindow(None, "Assetto Corsa")
        # if window_handle is not None:
        #     win32gui.ShowWindow(window_handle, win32con.SW_SHOWMINIMIZED)
        #     win32gui.ShowWindow(window_handle, win32con.SW_SHOWNORMAL)
        # else:
        #     print("Window not found")

        # use pyautogui to click on location instead of bringing window to front
        pyautogui.click(x=250, y=25)

    def rewind_time(self, seconds=5):
        
        game_seconds = seconds / 5

        self.focus_window()
        ctypes.windll.user32.keybd_event(0x4A, 0, KEYEVENTF_KEYDOWN, 0)
        # print(game_seconds)
        # print(seconds)
        sleep(game_seconds)
        ctypes.windll.user32.keybd_event(0x4A, 0, KEYEVENTF_KEYUP, 0)


    def close(self):
        self.gamepad.reset()
        self.gamepad.update()
        del self.gamepad


if __name__ == "__main__":

    game_controller = GameController()

    # Testing joypad
    # game_controller.set_inputs(0, 0, 0)
    # for i in np.arange(0,1.01,0.01):
    #     game_controller.set_inputs(0, i, i)
    #     sleep(0.05)

    # sleep(2)
    # print("Resetting inputs")
    # game_controller.reset_inputs()

    # sleep(2)
    # print("Pausing game")
    # game_controller.pause_game()
    # sleep(1)
    # print("Unpausing game")
    # game_controller.pause_game()
    # sleep(2)

    # print("Resetting game")
    # game_controller.restart_session()
    # sleep(2)

    # print("Accelerating!!")
    # game_controller.set_inputs(0,0,1)
    # sleep(3)

    # game_controller.close()

    # Testing keypresses

    # use pyautogui to get mouse position
    # sleep(5)
    # print(pyautogui.position())
    

    
    for i in range(5):
        print(f"{5-i}...")
        sleep(1)

    game_controller.rewind_time(rewind_time=5)