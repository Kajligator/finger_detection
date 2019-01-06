import cv2 as cv
import numpy as np
import math


# optional argument for trackbars
def nothing(x):
    pass

'''
VARIABLES
'''
THRESHOLD = 60
FONT_FACE = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_SCALE = 2
FONT_THICKNESS = 3

'''
MATH
'''
HALF_PI = math.pi / 2


MIN_HSV = np.array([0, 0, 255])
MAX_HSV = np.array([179, 65, 255])

camera = cv.VideoCapture(0)
camera.set(10, 200)

kernel = np.ones((3, 3), np.uint8)

# named ites for easy reference
barsWindow = 'Bars'
hl = 'H Low'
hh = 'H High'
sl = 'S Low'
sh = 'S High'
vl = 'V Low'
vh = 'V High'

# create window for the slidebars
cv.namedWindow(barsWindow, flags=cv.WINDOW_AUTOSIZE)

# create the sliders
cv.createTrackbar(hl, barsWindow, 0, 179, nothing)
cv.createTrackbar(hh, barsWindow, 0, 179, nothing)
cv.createTrackbar(sl, barsWindow, 0, 255, nothing)
cv.createTrackbar(sh, barsWindow, 0, 255, nothing)
cv.createTrackbar(vl, barsWindow, 0, 255, nothing)
cv.createTrackbar(vh, barsWindow, 0, 255, nothing)

# set initial values for sliders
cv.setTrackbarPos(hl, barsWindow, MIN_HSV[0])
cv.setTrackbarPos(hh, barsWindow, MAX_HSV[0])
cv.setTrackbarPos(sl, barsWindow, MIN_HSV[1])
cv.setTrackbarPos(sh, barsWindow, MAX_HSV[1])
cv.setTrackbarPos(vl, barsWindow, MIN_HSV[2])
cv.setTrackbarPos(vh, barsWindow, MAX_HSV[2])

while camera.isOpened():
    _, frame = camera.read()

    # read trackbar positions for all
    hul = cv.getTrackbarPos(hl, barsWindow)
    huh = cv.getTrackbarPos(hh, barsWindow)
    sal = cv.getTrackbarPos(sl, barsWindow)
    sah = cv.getTrackbarPos(sh, barsWindow)
    val = cv.getTrackbarPos(vl, barsWindow)
    vah = cv.getTrackbarPos(vh, barsWindow)

    # make array for final values
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, HSVLOW, HSVHIGH)
    dilation = cv.dilate(mask, kernel, iterations=1)

    ret, thresh = cv.threshold(dilation, THRESHOLD, 255, cv.THRESH_BINARY)

    # Check for contours and pick the largest one
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    length_of_contours = len(contours)
    largest_area = -1
    if length_of_contours > 0:
        largest_contour = None
        for i in range(length_of_contours):
            contour = contours[i]
            area = cv.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = i

        res = contours[largest_contour]

        hull = cv.convexHull(res, returnPoints=False)

        if len(hull) > 2:
            convex_defects = cv.convexityDefects(res, hull)
            if type(convex_defects) != type(None):
                valid_points = []
                x, y, w, h = cv.boundingRect(res)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 0), 2)
                center_x = (x + w) // 2
                center_y = (y + h) // 2
                # print(center_x, center_y)
                arr = []
                for i in range(convex_defects.shape[0]):
                    s, e, f, d = convex_defects[i][0]
                    point1 = tuple(res[s][0])
                    point2 = tuple(res[e][0])
                    point3 = tuple(res[f][0])

                    a = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                    b = math.sqrt((point3[0] - point1[0]) ** 2 + (point3[1] - point1[1]) ** 2)
                    c = math.sqrt((point2[0] - point3[0]) ** 2 + (point2[1] - point3[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                    add = math.radians(40)
                    add2 = math.radians(10)

                    if HALF_PI - add <= angle < HALF_PI:
                        valid_points.append(point1)
                        cv.putText(frame, str(angle), point1, FONT_FACE, 1 / 2, (128, 255, 64), 1, cv.LINE_AA)

                    if angle <= HALF_PI - add2:
                        valid_points.append(point2)
                        cv.putText(frame, str(angle), point2, FONT_FACE, 1 / 2, (128, 255, 64), 1, cv.LINE_AA)
                    arr.append(len(valid_points))
                most_common = np.bincount(arr).argmax()

                length_of_valid_points = len(valid_points)
                for i in range(length_of_valid_points):
                    cv.circle(frame, valid_points[i], 3, (255, 0, 0), 3)
                    # print(length_of_valid_points)
                    # cv.putText(frame, str(valid_points[i]), (valid_points[i]), FONT_FACE, 1/2, (128, 255, 64), 1, cv.LINE_AA)
                cv.putText(frame, str(most_common), (10, 200), FONT_FACE, FONT_SCALE, (255, 128, 0), FONT_THICKNESS, cv.LINE_AA)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    # cv.imshow('res', dilation)
    # cv.imshow('thresh', thresh)

    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
