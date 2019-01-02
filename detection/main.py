import cv2 as cv
import numpy as np
import math

'''
VARIABLES
'''
THRESHOLD = 60
FONT_FACE = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_SCALE = 2
FONT_THICKNESS = 3

kernel = np.ones((5, 5), np.uint8)


# MIN_HSV = np.array([0, 0, 221])
# MAX_HSV = np.array([29, 114, 255])
MIN_HSV = np.array([0, 0, 221])
MAX_HSV = np.array([179, 114, 255])

camera = cv.VideoCapture(0)
camera.set(10, 200)

kernel = np.ones((3, 3), np.uint8)

def calculate_inner_angle(px1, py1, px2, py2, cx1, cy1):

    dist1 = math.sqrt((px1 - cx1) * (px1 - cx1) + (py1 - cy1) * (py1 - cy1))
    dist2 = math.sqrt((px2 - cx1) * (px2 - cx1) + (py2 - cy1) * (py2 - cy1))

    Cx = cx1
    Cy = cy1
    if dist1 < dist2:
        Bx = px1
        By = py1
        Ax = px2
        Ay = py2
    else:
        Bx = px2
        By = py2
        Ax = px1
        Ay = py1

    Q1 = Cx - Ax
    Q2 = Cy - Ay
    P1 = Bx - Ax
    P2 = By - Ay

    A = math.acos((P1 * Q1 + P2 * Q2) / (math.sqrt(P1 * P1 + P2 * P2) * math.sqrt(Q1 * Q1 + Q2 * Q2) ) )

    A = A * 180 / math.pi

    return A


while camera.isOpened():
    _, frame = camera.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dilation = cv.dilate(hsv, kernel, iterations=1)
    mask = cv.inRange(dilation, MIN_HSV, MAX_HSV)
    ret, thresh = cv.threshold(mask, THRESHOLD, 255, cv.THRESH_BINARY)

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
        hull = cv.convexHull(res)

        # Draw on the main frame the larges contour
        # cv.drawContours(frame, [res], 0, (0, 255, 255), 2)
        # cv.drawContours(frame, [hull], 0, (0, 128, 128), 3)
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
                for i in range(convex_defects.shape[0]):
                    s, e, f, d = convex_defects[i][0]
                    point1 = tuple(res[s][0])
                    point2 = tuple(res[e][0])
                    point3 = tuple(res[f][0])
                    angle = math.atan2(center_y - point1[1], center_x - point1[0]) * 180 / math.pi
                    inAngle = calculate_inner_angle(point1[0], point1[1], point2[0], point2[1], point3[0], point3[1])
                    length = math.sqrt(math.pow(point1[0] - point3[0], 2) + math.pow(point1[0] - point3[1], 2))
                    if not (not (angle > -30) or not (angle < 160) or not (20 < math.fabs(inAngle) < 120)) and length > 0.1 * h:
                        valid_points.append(point1)

                    # cv.line(frame, point1, point3, (64, 255, 64), 3)
                    # cv.line(frame, point3, point2, (255, 128, 64), 3)
                length_of_valid_points = len(valid_points)
                for i in range(length_of_valid_points):
                    cv.circle(frame, valid_points[i], 10, (255, 0, 0), 3)
                    print(length_of_valid_points)
                    cv.putText(frame, str(length_of_valid_points), (10, 200), FONT_FACE, FONT_SCALE, (255, 128, 0), FONT_THICKNESS, cv.LINE_AA)

    cv.imshow('frame', frame)
    # cv.imshow('mask', mask)
    # cv.imshow('res', dilation)
    # cv.imshow('thresh', thresh)

    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
