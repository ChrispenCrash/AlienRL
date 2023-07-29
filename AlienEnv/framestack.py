import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from collections import deque
import threading
# from game_state import GameState
from time import sleep
from AlienEnv.utils import move_window_to_top_left
import pyautogui # Very important for window resize to work properly
from AlienEnv.config import GAME_FPS, WIDTH, HEIGHT, GAME_WINDOW_NAME



# gamestate = GameState()

def get_frame():

        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, WIDTH, HEIGHT)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (WIDTH, HEIGHT), srcdc, (0, 45), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")
        img.shape = (HEIGHT, WIDTH, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

class FrameStack:
    def __init__(self, framestack_skip=4, framestack_size=4):

        # Resizing issue fixed, pyautogui must be imported
        move_window_to_top_left(x=-5, y=0, width=WIDTH, height=HEIGHT) 
        
        self.deque_size = framestack_skip * (framestack_size - 1) + 1
        self.frames = deque(maxlen=self.deque_size)

        self.framestack_skip = framestack_skip
        self.frame_time = (1 / GAME_FPS)

        self.lock = threading.Lock()

        self.stop_event = threading.Event()
        self.frame_available = threading.Event()
        self.framestack_full = threading.Event()

        self.frame_counter = 0

    def start(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            frame = get_frame()
            with self.lock:
                self.frames.appendleft(frame)
                self.frame_counter += 1
                self.frame_available.set()
                if len(self.frames) == self.deque_size:
                    self.framestack_full.set()  # Signal that the framestack is full
            sleep(self.frame_time)  # wait for 1/60 seconds (roughly every frame)

    def get_latest_frame(self):
        # Ensure at least one frame in the queue
        self.frame_available.wait()
        with self.lock:
            return np.array(self.frames)[0]

    def get_framestack(self):
        self.framestack_full.wait()  # Wait until the framestack is full
        with self.lock:
            return np.array(self.frames)[0::self.framestack_skip]
        
    @property
    def frame_count(self):
        return self.frame_counter

    def stop(self):
        self.stop_event.set()
        self.thread.join()
