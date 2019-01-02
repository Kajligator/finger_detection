import cv2 as cv
import numpy as np
import math

'''
VARIABLES
'''
THRESHOLD = 60

# MIN_HSV = np.array([0, 0, 221])
# MAX_HSV = np.array([29, 114, 255])
MIN_HSV = np.array([0, 0, 221])
MAX_HSV = np.array([179, 114, 255])



element_size = 5
camera = cv.VideoCapture(0)
camera.set(10, 200)

kernel = np.ones((3, 3), np.uint8)

while camera.isOpened():
    _, frame = camera.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # dilation = cv.dilate(hsv, kernel, iterations=1)

    mask = cv.inRange(hsv, MIN_HSV, MAX_HSV)

    # res = cv.bitwise_and(frame, frame, mask=mask)

    # element = cv.getStructuringElement(cv.MORPH_ELLIPSE,
    #                                    (2 * element_size + 1, 2 * element_size + 1), (element_size, element_size))

    ret, thresh = cv.threshold(mask, THRESHOLD, 255, cv.THRESH_BINARY)
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    largest_area = -1
    if length > 0:
        largest_contour = None
        for i in range(length):
            temp = contours[i]
            area = cv.contourArea(temp)
            print(temp)
            if area > largest_area:
                largest_area = area
                largest_contour = i

        res = contours[largest_contour]
        draw = np.zeros(frame.shape, np.uint8)
        cv.drawContours(frame, [res], 0, (0, 255, 0), 3)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    # cv.imshow('res', dilation)
    cv.imshow('thresh', thresh)

    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
