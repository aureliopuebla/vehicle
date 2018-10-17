#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import random
import cv2

MAX_DISPARITY = 16383
HISTOGRAM_BINS = 256
BIN_SIZE = (MAX_DISPARITY+1) / HISTOGRAM_BINS
FLATNESS_THRESHOLD = 2
RANSAC_TRIES = 1000
RANSAC_EPSILON = 1
ROAD_LINE_FIT_ALPHA = 0.26
bridge = CvBridge()


def getHistogram(array):
    return np.histogram(array, bins=HISTOGRAM_BINS, range=(0, MAX_DISPARITY))[0]


def getUDisparityThressholdFilter(dispImage):
    rows, cols = dispImage.shape
    UdispImage = np.apply_along_axis(getHistogram, 0, dispImage)
    return UdispImage[np.minimum(HISTOGRAM_BINS-1, dispImage / BIN_SIZE),
                      np.mgrid[0:rows, 0:cols][1]] < FLATNESS_THRESHOLD


def getRANSACFittedLine(VdispImage):
    rows, cols = VdispImage.shape
    cumSumArray = np.cumsum(VdispImage)
    N = cumSumArray[-1]
    bestM = bestB = bestF = 0
    for i in range(RANSAC_TRIES):
        idx1 = np.searchsorted(cumSumArray, random.randint(1, N), side='left')
        idx2 = np.searchsorted(cumSumArray, random.randint(1, N), side='left')

        y1 = idx1 / cols
        x1 = idx1 - y1 * cols
        y2 = idx2 / cols
        x2 = idx2 - y2 * cols
        if x1 == x2: continue  # Do not consider vertical lines
        m = float(y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        f = 0
        for x in range(cols):
            y = int(m * x + b)
            if y < 0 or y >= rows: break
            for yp in range(
                max(0, y - RANSAC_EPSILON), min(rows, y + RANSAC_EPSILON)):
                f += VdispImage[yp][x]
        if f > bestF:
            bestF = f
            bestM = m
            bestB = b

    return bestM, bestB


def getRoadLineFitFilter(dispImage, m, b):
    m /= BIN_SIZE  # Adjust m to original disparity values.
    rows, _ = dispImage.shape
    roadRowValues = np.fromfunction(
        np.vectorize(lambda r, _: float(r - b) / m), (rows,1))
    return np.abs(
        dispImage - roadRowValues) <= (ROAD_LINE_FIT_ALPHA * roadRowValues)


def roadCallback(cameraImageMsg, dispImageMsg, VdispImagePub, roadImagePub):
    dispImage = bridge.imgmsg_to_cv2(
        dispImageMsg, desired_encoding='passthrough')
    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough')

    UDispFilter = getUDisparityThressholdFilter(dispImage)
    VDispImage = np.apply_along_axis(getHistogram, 1, dispImage * UDispFilter)
    m, b = getRANSACFittedLine(VDispImage)
    roadFilter = getRoadLineFitFilter(dispImage, m, b)


    # Show elements with values > 0.
    VDispImage = VDispImage.astype('uint8') * 255
    colorVD = cv2.cvtColor(VDispImage, cv2.COLOR_GRAY2RGB)
    cv2.line(colorVD, (0, int(b)), (100, int(100*m+b)), (0, 0, 255), 2)

    roadImage = cameraImage * roadFilter[:, :, np.newaxis]
    cv2.line(roadImage,
             (0, int(b)),
             (cameraImage.shape[1]-1, int(b)),
             (0, 0, 255),
             2)

    VdispImagePub.publish(bridge.cv2_to_imgmsg(colorVD, encoding='bgr8'))
    roadImagePub.publish(bridge.cv2_to_imgmsg(roadImage, encoding='bgr8'))


def listener():
    rospy.init_node('road', anonymous=False)

    VdispImagePub = rospy.Publisher('/camera/Vdisp', Image, queue_size=1)
    roadImagePub = rospy.Publisher('/camera/road', Image, queue_size=1)

    cameraImageSub = message_filters.Subscriber('/camera/image', Image)
    dispImageSub = message_filters.Subscriber('/camera/disp', Image)
    ts = message_filters.TimeSynchronizer([cameraImageSub, dispImageSub], 1)
    ts.registerCallback(roadCallback, VdispImagePub, roadImagePub)

    rospy.spin()


if __name__ == '__main__':
    listener()
