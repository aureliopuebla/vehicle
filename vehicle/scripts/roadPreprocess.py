#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import random
import cv2

# The following publications are for visualization only.
PUBLISH_UDISPARITY_ROAD_FILTER = False
PUBLISH_VDISPARITY_WITH_FITTED_LINE = False
PUBLISH_LINE_FITTED_ROAD = False
PUBLISH_CLOUD_COLORING = True

# Road Line Fit Parameters
# Note: Disparity values are uint16.
MAX_DISPARITY = 16383
HISTOGRAM_BINS = 256
BIN_SIZE = (MAX_DISPARITY+1) / HISTOGRAM_BINS
FLATNESS_THRESHOLD = 2
RANSAC_TRIES = 1000
RANSAC_EPSILON = 1
ROAD_LINE_FIT_ALPHA = 0.20

# Vanishing Point Detection Parameters
VP_N = 4  # Implementation Specific
VP_CANDIDATES_BOX_HEIGHT = 40
VP_LAMBDA = 4 * np.sqrt(2)
VP_KERNEL_SIZE = int(10 * VP_LAMBDA / np.pi) + 1  # Must be odd
VP_W0 = 2 * np.pi / VP_LAMBDA
VP_K = np.pi / 2
VP_DELTA = -VP_W0**2 / (VP_K**2 * 8)


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


def evaluateRANSACTry(VdispImage, m, b):
    """Tests how fit a certain RANSAC try is in fitting the road plane."""
    rows, cols = VdispImage.shape
    f = 0
    for x in range(cols):
        y = int(m * x + b)
        if y < 0 or y >= rows: break
        for yp in range(
                max(0, y - RANSAC_EPSILON), min(rows, y + RANSAC_EPSILON)):
            f += VdispImage[yp][x]
    return f


def getRANSACFittedLine(VdispImage):
    """Applies RANSAC to find the best line fit of the VDispImage. This is the
       line that fits the approximate road."""
    rows, cols = VdispImage.shape
    cumSumArray = np.cumsum(VdispImage)
    N = cumSumArray[-1]
    global bestM, bestB
    bestM *= BIN_SIZE  # Adjust m to VdispImage dimensions.
    bestF = evaluateRANSACTry(VdispImage, bestM, bestB)
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
        f = evaluateRANSACTry(VdispImage, m, b)

        if f > bestF:
            bestF = f
            bestM = m
            bestB = b
    bestM /= BIN_SIZE  # Adjust m to original dispImage dimensions.
    return bestM, bestB


def getRoadLineFitFilter(dispImage, m, b):
    """Returns a boolean filter for the original road image of the road values
       that are close to the fitted road line."""
    rows, _ = dispImage.shape
    roadRowValues = np.fromfunction(
        np.vectorize(lambda r, _: float(r - b) / m), (rows,1))
    return np.abs(
        dispImage - roadRowValues) <= ROAD_LINE_FIT_ALPHA * roadRowValues


def applyGaborKernels(cameraImage, b, gaborKernels):
    top = int(b - VP_CANDIDATES_BOX_HEIGHT / 4)
    bottom = int(b + VP_CANDIDATES_BOX_HEIGHT * 3 / 4)


def preprocessRoadCallback(cameraImageMsg,
                           dispImageMsg,
                           bridge,
                           unused_gaborKernels,
                           UdispRoadFilterImagePub=None,
                           VdispWithFittedLineImagePub=None,
                           lineFittedRoadImagePub=None,
                           cloudColoringImagePub=None):
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
      cloudColoringImagePub: If set, it-s the ROS Publisher that contains the
        corresponding colors for the point cloud based on a color code where
        RED are obstacles and BLUE is the detected road.
    """
    dispImage = bridge.imgmsg_to_cv2(
        dispImageMsg, desired_encoding='passthrough')
    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough')

    UDispFilter = getUDisparityThressholdFilter(dispImage)
    VDispImage = np.apply_along_axis(getHistogram, 1, dispImage * UDispFilter)
    m, b = getRANSACFittedLine(VDispImage)
    lineFittedRoadFilter = getRoadLineFitFilter(dispImage, m, b)
    # applyGaborKernels(cameraImage, b, *gaborKernels)

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
        lineFittedRoadImagePub.publish(
            bridge.cv2_to_imgmsg(lineFittedRoad, encoding='bgr8'))

    if cloudColoringImagePub is not None:
        cloudColoring = 255 * np.ones(cameraImage.shape, np.uint8)
        cloudColoring[:, :, 0:1] *= lineFittedRoadFilter[:, :, np.newaxis]
        cloudColoringImageMsg = bridge.cv2_to_imgmsg(
            cloudColoring, encoding='bgr8')
        cloudColoringImageMsg.header = cameraImageMsg.header
        cloudColoringImagePub.publish(cloudColoringImageMsg)


def getGaborFilterKernels():
    gaborKernels = np.zeros(
        (VP_KERNEL_SIZE, VP_KERNEL_SIZE, VP_N), dtype=np.complex128)
    for i in range(VP_N):
        theta = np.pi/2 + i*np.pi/VP_N
        for y in range(-VP_KERNEL_SIZE//2, VP_KERNEL_SIZE//2+1):
            ySinTheta = y * np.sin(theta)
            yCosTheta = y * np.cos(theta)
            for x in range(-VP_KERNEL_SIZE//2, VP_KERNEL_SIZE//2+1):
                xCosTheta = x * np.cos(theta)
                xSinTheta = x * np.sin(theta)
                a = xCosTheta + ySinTheta
                b = -xSinTheta + yCosTheta
                gaborKernels[y+VP_KERNEL_SIZE//2, x+VP_KERNEL_SIZE//2, i] = (
                        VP_W0 / (np.sqrt(2 * np.pi) * VP_K) *
                        np.exp(VP_DELTA * (4 * a**2 + b**2)) *
                        (np.exp(1j * VP_W0 * a) - np.exp(-VP_K**2 / 2)))
    np.save('gaborKernels.npy', gaborKernels)
    raise Exception('DONE')
    return gaborKernels


def listener():
    rospy.init_node('roadPreprocess', anonymous=False)
    bridge = CvBridge()

    global bestM, bestB
    bestM = rospy.get_param('~initial_M', 0.0)
    bestB = rospy.get_param('~initial_B', 0.0)

    UdispRoadFilterImagePub = (
        rospy.Publisher('/camera/UdispRoadFilter/image', Image, queue_size=1)
        if PUBLISH_UDISPARITY_ROAD_FILTER else None)
    VdispWithFittedLineImagePub = (
        rospy.Publisher('/camera/VdispWithFittedLine/image', Image, queue_size=1)
        if PUBLISH_VDISPARITY_WITH_FITTED_LINE else None)
    lineFittedRoadImagePub = (
        rospy.Publisher('/camera/lineFittedRoad/image', Image, queue_size=1)
        if PUBLISH_LINE_FITTED_ROAD else None)
    cloudColoringImagePub = (
        rospy.Publisher('/camera/cloudColoring/image', Image, queue_size=1)
        if PUBLISH_CLOUD_COLORING else None)
    # TODO: Publish roadLinePub and vanishingPointPub.

    cameraImageSub = message_filters.Subscriber('/camera/left/image_rect',
                                                Image)
    dispImageSub = message_filters.Subscriber('/camera/disp/image_rect', Image)
    ts = message_filters.TimeSynchronizer([cameraImageSub, dispImageSub], 1)
    ts.registerCallback(preprocessRoadCallback,
                        bridge,
                        None, #getGaborFilterKernels(),
                        UdispRoadFilterImagePub,
                        VdispWithFittedLineImagePub,
                        lineFittedRoadImagePub,
                        cloudColoringImagePub)

    rospy.spin()


if __name__ == '__main__':
    listener()
