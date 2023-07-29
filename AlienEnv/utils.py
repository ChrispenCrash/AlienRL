import win32gui
import win32con
import pyautogui # Very important
from AlienEnv.config import GAME_WINDOW_NAME

def move_window_to_top_left(x=0, y=0, width=1280, height=720, window_name='Assetto Corsa'):
    # Find the window by its name
    window_handle = win32gui.FindWindow(None, window_name)

    left_padding = 11
    missing_width = 10 # No clue why but this is needed
    missing_height = 11 # No clue why but this is needed

    # If the window is found
    if window_handle:
        # Move and resize the window
        # Parameters: window handle, x, y, width, height
        win32gui.MoveWindow(window_handle, 0 - left_padding, 0, width + missing_width + 12, height + 45 + missing_height, True)
    else:
        print(f'Window - {window_name} - not found')