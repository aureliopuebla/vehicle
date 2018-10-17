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
bridge = CvBridge()


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


def getHistogram(array):
    return np.histogram(
        array, bins=HISTOGRAM_BINS, range=(0, MAX_DISPARITY))[0]


def getRoadThressholdFilter(dispImage):
    rows, cols = dispImage.shape
    UdispImage = np.apply_along_axis(getHistogram, 0, dispImage)
    return UdispImage[np.minimum(HISTOGRAM_BINS-1, dispImage / BIN_SIZE),
                      np.mgrid[0:rows, 0:cols][1]] < FLATNESS_THRESHOLD


def roadCallback(cameraImageMsg, dispImageMsg, VdispImagePub, roadImagePub):
    dispImage = bridge.imgmsg_to_cv2(
        dispImageMsg, desired_encoding='passthrough')
    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough')

    filterImage = getRoadThressholdFilter(dispImage)
    VdispImage = np.apply_along_axis(getHistogram, 1, dispImage * filterImage)
    m, b = getRANSACFittedLine(VdispImage)

    # Show elements with values > 0.
    VdispImage = VdispImage.astype('uint8') * 255
    colorVD = cv2.cvtColor(VdispImage, cv2.COLOR_GRAY2RGB)
    cv2.line(colorVD, (0, int(b)), (100, int(100*m+b)), (255, 0, 0), 2)
    filterImage = filterImage.astype('uint16') * 65535

    VdispImagePub.publish(bridge.cv2_to_imgmsg(
        colorVD, encoding='bgr8'))
    roadImagePub.publish(bridge.cv2_to_imgmsg(
        filterImage.astype('uint16'), encoding='16UC1'))


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
