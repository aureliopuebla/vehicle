#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import random
import cv2

# The following publications are for visualization only.
PUBLISH_UDISPARITY_ROAD_FILTER = True
PUBLISH_VDISPARITY_WITH_FITTED_LINE = True
PUBLISH_LINE_FITTED_ROAD = True

# Note: Disparity values are uint16.
MAX_DISPARITY = 16383
HISTOGRAM_BINS = 256
BIN_SIZE = (MAX_DISPARITY+1) / HISTOGRAM_BINS
FLATNESS_THRESHOLD = 2
RANSAC_TRIES = 2000
RANSAC_EPSILON = 1
ROAD_LINE_FIT_ALPHA = 0.20
VANISHING_POINT_CANDIDATES_BOX_HEIGHT = 40
VANISHING_POINT_CANDIDATES_BOX_WIDTH = 300


def getHistogram(array):
    """Given an array [a0, a1, ...], return a histogram with 'HISTOGRAM_BINS'
       bins in range 0 to MAX_DISPARITY. Values out of range are ignored."""
    return np.histogram(array, bins=HISTOGRAM_BINS, range=(0, MAX_DISPARITY))[0]


def getUDisparityThressholdFilter(dispImage):
    """Calculates the UDisparity from the given dispImage, and uses it to return
       a boolean filter for dispImage where 'True' is assigned to a given (r,c)
       coordinate iff its corresponding UDisparity value is below a given
       FLATNESS_THRESSHOLD. Such UDisparity value is given by the frequency
       value that corresponds to the disparity value (r,c) in the histogram of
       column 'c'."""
    rows, cols = dispImage.shape
    UdispImage = np.apply_along_axis(getHistogram, 0, dispImage)
    return UdispImage[np.minimum(HISTOGRAM_BINS-1, dispImage / BIN_SIZE),
                      np.mgrid[0:rows, 0:cols][1]] < FLATNESS_THRESHOLD


def getRANSACFittedLine(VdispImage):
    """Applies RANSAC to find the best line fit of the VDispImage. This is the
       line that fits the approximate road."""
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
    bestM /= BIN_SIZE  # Adjust m to original disparity values.
    return bestM, bestB


def getRoadLineFitFilter(dispImage, m, b):
    """Returns a boolean filter for the original road image of the road values
       that are close to the fitted road line."""
    rows, _ = dispImage.shape
    roadRowValues = np.fromfunction(
        np.vectorize(lambda r, _: float(r - b) / m), (rows,1))
    return np.abs(
        dispImage - roadRowValues) <= ROAD_LINE_FIT_ALPHA * roadRowValues


def getVanishingPointCandidatesBox(lineFittedRoadFilter, b):
    top = int(b - VANISHING_POINT_CANDIDATES_BOX_HEIGHT / 4)
    bottom = int(b + VANISHING_POINT_CANDIDATES_BOX_HEIGHT * 3 / 4)
    _, cols = lineFittedRoadFilter.shape
    VPCandidateXCoordanates = (lineFittedRoadFilter[top:bottom+1] *
                               np.mgrid[0:bottom-top+1, 0:cols][1])
    Xmean = (np.sum(VPCandidateXCoordanates) /
             np.count_nonzero(lineFittedRoadFilter[top:bottom+1]))
    left = int(max(0, Xmean - VANISHING_POINT_CANDIDATES_BOX_WIDTH/2))
    right = int(min(cols-1, Xmean + VANISHING_POINT_CANDIDATES_BOX_WIDTH/2))
    return left, top, right, bottom


def preprocessRoadCallback(cameraImageMsg,
                           dispImageMsg,
                           bridge,
                           UdispRoadFilterImagePub=None,
                           VdispWithFittedLineImagePub=None,
                           lineFittedRoadImagePub=None):
    """Prepocesses the Road with the help of the dispImage.

    Args:
      cameraImageMsg: A ROS Message containing the color Road Image.
      dispImageMsg: A ROS Message containing the corresponding disparity Image.
      bridge: The CV bridge instance used to convert CV images and ROS Messages.
      UdispRoadFilterImagePub: If set, it's the ROS Publisher that will contain
        the UDisparity Filter for visualization.
      VdispWithFittedLineImagePub: If set, it's the ROS Publisher that will
        contain the VDisparity With the RANSAC fitted line for visualization.
      lineFittedRoadImagePub: If set, it's the ROS Publisher that will contain
        the Line Fitted Road Image with horizon line for visualization.
    """
    dispImage = bridge.imgmsg_to_cv2(
        dispImageMsg, desired_encoding='passthrough')
    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough')

    UDispFilter = getUDisparityThressholdFilter(dispImage)
    VDispImage = np.apply_along_axis(getHistogram, 1, dispImage * UDispFilter)
    m, b = getRANSACFittedLine(VDispImage)
    lineFittedRoadFilter = getRoadLineFitFilter(dispImage, m, b)
    left, top, right, bottom = getVanishingPointCandidatesBox(
        lineFittedRoadFilter, b)
    # https: // ieeexplore.ieee.org / document / 5957282

    if UdispRoadFilterImagePub is not None:
        # Convert Binary Image to uint8.
        UdispRoadFilterImagePub.publish(bridge.cv2_to_imgmsg(
            UDispFilter.astype('uint8')*255, encoding='8UC1'))

    if VdispWithFittedLineImagePub is not None:
        # Show elements with values > 0.
        VDispImage = VDispImage.astype('uint8')*255
        VdispWithFittedLine = cv2.cvtColor(VDispImage, cv2.COLOR_GRAY2RGB)
        _, cols = VDispImage.shape
        cv2.line(VdispWithFittedLine,
                 (0, int(b)),
                 (cols-1, int((cols-1) * (m*BIN_SIZE) + b)),
                 (0, 0, 255),
                 2)
        VdispWithFittedLineImagePub.publish(
            bridge.cv2_to_imgmsg(VdispWithFittedLine, encoding='bgr8'))

    if lineFittedRoadImagePub is not None:
        lineFittedRoad = cameraImage * lineFittedRoadFilter[:, :, np.newaxis]
        cv2.line(lineFittedRoad,
                 pt1=(0, int(b)),
                 pt2=(cameraImage.shape[1] - 1, int(b)),
                 color=(0, 0, 255),
                 thickness=2)
        cv2.rectangle(lineFittedRoad,
                      pt1=(left, top),
                      pt2=(right, bottom),
                      color=(255, 0, 0),
                      thickness=2)
        lineFittedRoadImagePub.publish(
            bridge.cv2_to_imgmsg(lineFittedRoad, encoding='bgr8'))


def listener():
    rospy.init_node('roadPreprocess', anonymous=False)
    bridge = CvBridge()

    UdispRoadFilterImagePub = (
        rospy.Publisher('/camera/UdispRoadFilter', Image, queue_size=1)
        if PUBLISH_UDISPARITY_ROAD_FILTER else None)
    VdispWithFittedLineImagePub = (
        rospy.Publisher('/camera/VdispWithFittedLine', Image, queue_size=1)
        if PUBLISH_VDISPARITY_WITH_FITTED_LINE else None)
    lineFittedRoadImagePub = (
        rospy.Publisher('/camera/lineFittedRoad', Image, queue_size=1)
        if PUBLISH_LINE_FITTED_ROAD else None)
    # TODO: Publish roadLinePub and vanishingPointPub.

    cameraImageSub = message_filters.Subscriber('/camera/image', Image)
    dispImageSub = message_filters.Subscriber('/camera/disp', Image)
    ts = message_filters.TimeSynchronizer([cameraImageSub, dispImageSub], 1)
    ts.registerCallback(preprocessRoadCallback,
                        bridge,
                        UdispRoadFilterImagePub,
                        VdispWithFittedLineImagePub,
                        lineFittedRoadImagePub)

    rospy.spin()


if __name__ == '__main__':
    listener()
