import time
import os
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from telemetry import TelemetryData
from typing import Tuple


class GameState:
    def __init__(self):
        pass

    def get_initial_observation(self):
        frames = self.get_framestack()
        telemetry = self.get_telemetry()

        return np.array(frames, telemetry)

    def get_frame(self, x=0, y=31, width=1280, height=720) -> np.ndarray:
        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (x, y), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")


        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        # https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # return img
    
    def roi(self, img, vertices):
        
        #blank mask:
        mask = np.zeros_like(img)   
        
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, 255)
        
        #returning the image only where mask pixels are nonzero
        masked = cv2.bitwise_and(img, mask)
        return masked
    
    def process_img(self, image):
        original_image = image
        # convert to gray
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # edge detection
        processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
        
        processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
        
        vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                            ], np.int32)

        # processed_img = self.roi(processed_img, [vertices])

        # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        #                                     rho   theta   thresh  min length, max gap:        
        lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,      20,       15)
        m1 = 0
        m2 = 0
        try:
            for coords in lines:
                coords = coords[0]
                try:
                    cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                    
                    
                except Exception as e:
                    print(str(e))
        except Exception:
            pass

        return processed_img,original_image, m1, m2
    
    def get_framestack(self) -> np.ndarray:
        pass

    def get_telemetry(self) -> np.ndarray:
        telemetry = TelemetryData()

        x, y, z = list(telemetry.graphics.carCoordinates)

        telemetry.close()

