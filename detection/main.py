import cv2 as cv
import numpy as np
import math

'''
VARIABLES
'''
THRESHOLD = 60

MIN_HSV = np.array([0, 0, 221])
MAX_HSV = np.array([29, 114, 255])
element_size = 5

camera = cv.VideoCapture(0)
camera.set(10,200)
# cv.namedWindow('trackbar')
# cv.createTrackbar('trh1', 'trackbar', THRESHOLD, 100, printThreshold)


while camera.isOpened():
    _, frame = camera.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, MIN_HSV, MAX_HSV)

    res = cv.bitwise_and(frame, frame, mask=mask)

    blur = cv.medianBlur(res, 5)
    kernel = np.ones((3, 3), np.uint8)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * element_size + 1, 2 * element_size + 1), (element_size, element_size))
    dilation = cv.dilate(blur, kernel, iterations=1)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', dilation)

    # cnt = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, (0, 0))
    largest_contour = 0
    # for i in len(cnt):
    # print(cnt)
    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
