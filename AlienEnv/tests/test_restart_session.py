from controller import GameController
from time import sleep
import pyautogui
import ctypes

KEYEVENTF_KEYDOWN = 0x0
KEYEVENTF_KEYUP = 0x2

game_controller = GameController()

# print(pyautogui.position())
# pyautogui.click(x=250, y=25)
# print(pyautogui.position())

# # Press ESC key to pause game
# ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYDOWN, 0)
# ctypes.windll.user32.keybd_event(0x1B, 0, KEYEVENTF_KEYUP, 0)

# pyautogui.click(x=645, y=426)
# sleep(5)
# pyautogui.click(x=49, y=217)
# sleep(2)

# game_controller.pause_game()
game_controller.restart_session()