import cv2 as cv
import numpy as np
import math

'''
VARIABLES
'''
THRESHOLD = 60

MIN_HSV = np.array([0, 0, 221])
MAX_HSV = np.array([29, 114, 255])

camera = cv.VideoCapture(0)
camera.set(10, 200)

kernel = np.ones((3, 3), np.uint8)

while camera.isOpened():
    _, frame = camera.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dilation = cv.dilate(hsv, kernel, iterations=1)

    mask = cv.inRange(dilation, MIN_HSV, MAX_HSV)

    res = cv.bitwise_and(frame, frame, mask=mask)

    blur = cv.medianBlur(res, 5)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * element_size + 1, 2 * element_size + 1), (element_size, element_size))

    ret, thresh = cv.threshold(blur, THRESHOLD, 255, cv.THRESH_BINARY)
    _, contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        print(cnt)

    cv.imshow('frame', frame)
    # cv.imshow('mask', mask)
    # cv.imshow('res', dilation)
    cv.imshow('thresh', thresh)

    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
