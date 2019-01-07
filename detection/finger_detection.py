import json

import cv2 as cv
import numpy as np
import math

with open('./hsv_settings.json') as file:
    hsv_settings = json.load(file)

'''
VARIABLES
'''
THRESHOLD = 65
FONT_FACE = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_SCALE = 2
FONT_THICKNESS = 3

HALF_PI = math.pi / 2
MIN_ANGLE_THUMB = HALF_PI - math.radians(40)    #40
MAX_ANGLE_FINGERS = HALF_PI - math.radians(5)  #10

# Default hsv maks settings
MIN_HSV = np.array([0, 121, 255])
MAX_HSV = np.array([36, 255, 255])

# named items for easy reference
barsWindow = 'Bars'
hl_name = 'H Low'
hh_name = 'H High'
sl_name = 'S Low'
sh_name = 'S High'
vl_name = 'V Low'
vh_name = 'V High'

# Give CV acces to camera
camera = cv.VideoCapture(0)
camera.set(10, 200)

kernel = np.ones((3, 3), np.uint8)

# create window for the slidebars
cv.namedWindow(barsWindow, flags=cv.WINDOW_AUTOSIZE)


# read trackbar positions for all
hul = hsv_settings['hl']
huh = hsv_settings['hh']
sal = hsv_settings['sl']
sah = hsv_settings['sh']
val = hsv_settings['vl']
vah = hsv_settings['vh']


# update setting for hsv
def update(x):
    print(x)
    hsv_settings_object = {
        "hl": hul,
        "hh": huh,
        "sl": sal,
        "sh": sah,
        "vl": val,
        "vh": vah
    }

    with open('./hsv_settings.json', 'w') as write_file:
        json.dump(hsv_settings_object, write_file)


def calculate_arm(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


# create the sliders
cv.createTrackbar(hl_name, barsWindow, 0, 179, update)
cv.createTrackbar(hh_name, barsWindow, 0, 179, update)
cv.createTrackbar(sl_name, barsWindow, 0, 255, update)
cv.createTrackbar(sh_name, barsWindow, 0, 255, update)
cv.createTrackbar(vl_name, barsWindow, 0, 255, update)
cv.createTrackbar(vh_name, barsWindow, 0, 255, update)

# set initial values for sliders
cv.setTrackbarPos(hl_name, barsWindow, hsv_settings['hl'])
cv.setTrackbarPos(hh_name, barsWindow, hsv_settings['hh'])
cv.setTrackbarPos(sl_name, barsWindow, hsv_settings['sl'])
cv.setTrackbarPos(sh_name, barsWindow, hsv_settings['sh'])
cv.setTrackbarPos(vl_name, barsWindow, hsv_settings['vl'])
cv.setTrackbarPos(vh_name, barsWindow, hsv_settings['vh'])

while camera.isOpened():
    _, frame = camera.read()

    # read trackbar positions for all
    hul = cv.getTrackbarPos(hl_name, barsWindow)
    huh = cv.getTrackbarPos(hh_name, barsWindow)
    sal = cv.getTrackbarPos(sl_name, barsWindow)
    sah = cv.getTrackbarPos(sh_name, barsWindow)
    val = cv.getTrackbarPos(vl_name, barsWindow)
    vah = cv.getTrackbarPos(vh_name, barsWindow)

    # make array for final values
    hsv_low = np.array([hul, sal, val])
    hsv_high = np.array([huh, sah, vah])
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, hsv_low, hsv_high)
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
        # hull = cv.convexHull(res)
        # cv.drawContours(frame, [hull], 0, (0, 0, 255), 3)

    # Create convexes
        hull = cv.convexHull(res, returnPoints=False)
    # Check if there are convexes
        if len(hull) > 2:
            convex_defects = cv.convexityDefects(res, hull)  # Find points of defections
            if type(convex_defects) != type(None):           # Sometimes convex defects are a None type  ¯\_(ツ)_/¯
                valid_points = []
                x, y, w, h = cv.boundingRect(res)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 0), 2)
                center_x = (x + w) // 2
                center_y = (y + h) // 2
                amount_of_valid_points_collector = []       # Collect all the amount of the found points
                for i in range(convex_defects.shape[0]):
                    s, e, f, d = convex_defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
    # Calculating the arms of the triangles
                    arm_a = calculate_arm(end, start)
                    arm_b = calculate_arm(far, start)
                    arm_c = calculate_arm(end, far)

    # Calculating the angle between the fingers
                    angle = math.acos((math.pow(arm_b, 2) + math.pow(arm_c, 2) - math.pow(arm_a, 2)) / (2 * arm_b * arm_c))

    # Check only for the thumb
                    if MIN_ANGLE_THUMB <= angle < HALF_PI:
                        valid_points.append(start)
                        cv.putText(frame, str(math.degrees(angle)), start, FONT_FACE, 0.6, (255, 128, 0),
                                   1, cv.LINE_AA)

    # Check only for the fingers
                    if angle <= MAX_ANGLE_FINGERS:
                        valid_points.append(end)
                        cv.putText(frame, str(math.degrees(angle)), end, FONT_FACE, 0.6, (128, 255, 0),
                                   1, cv.LINE_AA)

                    length_of_valid_points = len(valid_points)
                    amount_of_valid_points_collector.append(length_of_valid_points)

    # Get the most common amount of fingers, to make the detection less... untamed
                most_common = np.bincount(amount_of_valid_points_collector).argmax()
                cv.putText(frame, str(most_common), (10, 200), FONT_FACE, FONT_SCALE, (255, 128, 0), FONT_THICKNESS, cv.LINE_AA)
                length_of_valid_points = len(valid_points)

                for i in range(length_of_valid_points):
                    cv.circle(frame, valid_points[i], 3, (255, 0, 0), -1)
                    print("point: {}, Cor: {}".format(i, valid_points[i]))
    cv.imshow('mask', mask)
    cv.imshow('frame', frame)

    # press ESC to exit
    k = cv.waitKey(10)
    if k == 27:
        break
