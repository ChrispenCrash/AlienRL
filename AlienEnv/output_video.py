import cv2
from time import sleep
import numpy as np
from gamestate import GameState
import datetime
import keyboard
import win32gui
import win32ui
import win32con
import pyautogui
from utils import move_window_to_top_left
from torchvision import transforms
from config import OUTPUT_WINDOW_NAME

# Full screen
WIDTH, HEIGHT = (1280, 720)
PIXELS_FROM_LEFT = 0
PIXELS_FROM_TOP = 45 # 45 for 4k monitor, 31 otherwise

cv2.namedWindow(OUTPUT_WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(OUTPUT_WINDOW_NAME, WIDTH, HEIGHT)

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

move_window_to_top_left(x=-5, y=0, width=WIDTH, height=HEIGHT)

i = 0

gamestate = GameState()

# sleep(1)

while True:
    
    frame = gamestate.get_cv2_frame()

    cv2.resizeWindow(OUTPUT_WINDOW_NAME, WIDTH, HEIGHT)
            
    cv2.putText(frame, "Press 'q' to quit", (WIDTH - 150, HEIGHT - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    total_frames = total_frames + 1
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)
    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, f"{fps_text}", (WIDTH - 150, HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow(OUTPUT_WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        gamestate.close()
        break

    i += 1

    if i == 60:
        i = 0

gamestate.close()
cv2.destroyAllWindows()