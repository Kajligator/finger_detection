import cv2 as cv
import numpy as np
import math

'''
VARIABLES
'''
THRESHOLD = 60

# MIN_HSV = np.array([130, 10, 75])
# MAX_HSV = np.array([160, 40, 130])
MIN_HSV = np.array([0, 0, 221])
MAX_HSV = np.array([28, 114, 255])

camera = cv.VideoCapture(0)
camera.set(10,200)
# cv.namedWindow('trackbar')
# cv.createTrackbar('trh1', 'trackbar', THRESHOLD, 100, printThreshold)


while camera.isOpened():
    _, frame = camera.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, MIN_HSV, MAX_HSV)

    res = cv.bitwise_and(frame, frame, mask=mask)

    res = cv.medianBlur(res, 5)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
