#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from string import Template
import numpy as np
import cv2
import time

CUDA_VDISP_LINE_RANSAC_FITTER_FILENAME = 'vdisp_line_ransac_fitter.cu'


def get_histogram(array):
    """Given an array [a0, a1, ...], return a histogram with 'HISTOGRAM_BINS'
       bins in range 0 to MAX_DISPARITY. Values out of range are ignored."""
    return np.histogram(array, bins=HISTOGRAM_BINS, range=(0, MAX_DISPARITY))[0]


def get_udisp_thresshold_filter(disp_image):
    """Calculates the UDisparity from the given dispImage, and uses it to return
       a boolean filter for dispImage where 'True' is assigned to a given (r,c)
       coordinate iff its corresponding UDisparity value is below a given
       FLATNESS_THRESSHOLD. Such UDisparity value is given by the frequency
       value that corresponds to the disparity value (r,c) in the histogram of
       column 'c'."""
    rows, cols = disp_image.shape
    udisp_image = np.apply_along_axis(get_histogram, 0, disp_image)
    return udisp_image[np.minimum(HISTOGRAM_BINS-1, disp_image / BIN_SIZE),
                       np.mgrid[0:rows, 0:cols][1]] < FLATNESS_THRESHOLD


def get_road_line_fit_filter(disp_image, m, b):
    """Returns a boolean filter for the original road image of the road values
       that are close to the fitted road line."""
    rows, _ = disp_image.shape
    road_row_values = np.fromfunction(
        np.vectorize(lambda r, _: float(r - b) / m), (rows, 1))
    return np.abs(
        disp_image - road_row_values) <= ROAD_LINE_FIT_ALPHA * road_row_values


def get_ransac_fitted_vdisp_line(vdisp_image):
    vdisp_image = vdisp_image.astype(np.int32)
    rows, cols = vdisp_image.shape
    cum_sum_array = np.cumsum(vdisp_image, dtype=np.int32)
    N = cum_sum_array[-1]
    m = np.empty(1, dtype=np.float32)
    b = np.empty(1, dtype=np.float32)
    getVdispLine(
        cuda.In(vdisp_image),
        np.int32(rows),
        np.int32(cols),
        cuda.In(cum_sum_array),
        np.int32(N),
        cuda.Out(m),
        cuda.Out(b),
        block=(CUDA_THREADS, 1, 1))
    return m[0] / BIN_SIZE, b[0]


def apply_gabor_kernels(camera_image, b, gabor_kernels):
    top = int(b - VP_CANDIDATES_BOX_HEIGHT / 4)
    bottom = int(b + VP_CANDIDATES_BOX_HEIGHT * 3 / 4)


def preprocess_road_callback(camera_image_msg,
                             disp_image_msg,
                             cv_bridge,
                             unused_gabor_kernels,
                             udisp_road_filter_image_pub=None,
                             vdisp_with_fitted_line_image_pub=None,
                             line_fitted_road_image_pub=None,
                             cloud_coloring_image_pub=None):
    """Prepocesses the Road with the help of the dispImage.

    Args:
      camera_image_msg: A ROS Message containing the color Road Image.
      disp_image_msg: A ROS Message containing the corresponding disparity Image.
      cv_bridge: The CV bridge instance used to convert CV images and ROS Messages.
      udisp_road_filter_image_pub: If set, it's the ROS Publisher that will contain
        the UDisparity Filter for visualization.
      vdisp_with_fitted_line_image_pub: If set, it's the ROS Publisher that will
        contain the VDisparity With the RANSAC fitted line for visualization.
      line_fitted_road_image_pub: If set, it's the ROS Publisher that will contain
        the Line Fitted Road Image with horizon line for visualization.
      cloud_coloring_image_pub: If set, it-s the ROS Publisher that contains the
        corresponding colors for the point cloud based on a color code where
        RED are obstacles and BLUE is the detected road.
    """
    disp_image = cv_bridge.imgmsg_to_cv2(
        disp_image_msg, desired_encoding='passthrough')
    camera_image = cv_bridge.imgmsg_to_cv2(
        camera_image_msg, desired_encoding='passthrough')

    udisp_filter = get_udisp_thresshold_filter(disp_image)
    vdisp_image = np.apply_along_axis(
        get_histogram, 1, disp_image * udisp_filter)
    m, b = get_ransac_fitted_vdisp_line(vdisp_image)
    line_fitted_road_filter = get_road_line_fit_filter(disp_image, m, b)
    # applyGaborKernels(cameraImage, b, *gaborKernels)

    if udisp_road_filter_image_pub is not None:
        # Convert Binary Image to uint8.
        udisp_road_filter_image_pub.publish(cv_bridge.cv2_to_imgmsg(
            udisp_filter.astype('uint8')*255, encoding='8UC1'))

    if vdisp_with_fitted_line_image_pub is not None:
        # Show elements with values > 0.
        vdisp_image = vdisp_image.astype('uint8')*255
        vdisp_with_fitted_line = cv2.cvtColor(vdisp_image, cv2.COLOR_GRAY2RGB)
        _, cols = vdisp_image.shape
        cv2.line(vdisp_with_fitted_line,
                 (0, int(b)),
                 (cols-1, int((cols-1) * (m*BIN_SIZE) + b)),
                 (0, 0, 255),
                 2)
        vdisp_with_fitted_line_image_pub.publish(
            cv_bridge.cv2_to_imgmsg(vdisp_with_fitted_line, encoding='bgr8'))

    if line_fitted_road_image_pub is not None:
        line_fitted_road = camera_image * \
            line_fitted_road_filter[:, :, np.newaxis]
        cv2.line(line_fitted_road,
                 pt1=(0, int(b)),
                 pt2=(camera_image.shape[1] - 1, int(b)),
                 color=(0, 0, 255),
                 thickness=2)
        line_fitted_road_image_pub.publish(
            cv_bridge.cv2_to_imgmsg(line_fitted_road, encoding='bgr8'))

    if cloud_coloring_image_pub is not None:
        cloud_coloring = 255 * np.ones(camera_image.shape, np.uint8)
        cloud_coloring[:, :, 0:1] *= line_fitted_road_filter[:, :, np.newaxis]
        cloud_coloring_image_msg = cv_bridge.cv2_to_imgmsg(
            cloud_coloring, encoding='bgr8')
        cloud_coloring_image_msg.header = camera_image_msg.header
        cloud_coloring_image_pub.publish(cloud_coloring_image_msg)


def get_gabor_filter_kernels():
    gabor_kernels = np.zeros(
        (VP_KERNEL_SIZE, VP_KERNEL_SIZE, VP_N), dtype=np.complex128)
    for i in range(VP_N):
        theta = np.pi/2 + i*np.pi/VP_N
        for y in range(-VP_KERNEL_SIZE//2, VP_KERNEL_SIZE//2+1):
            y_sin_theta = y * np.sin(theta)
            y_cos_theta = y * np.cos(theta)
            for x in range(-VP_KERNEL_SIZE//2, VP_KERNEL_SIZE//2+1):
                x_cos_theta = x * np.cos(theta)
                x_sin_theta = x * np.sin(theta)
                a = x_cos_theta + y_sin_theta
                b = -x_sin_theta + y_cos_theta
                gabor_kernels[y+VP_KERNEL_SIZE//2, x+VP_KERNEL_SIZE//2, i] = (
                    VP_W0 / (np.sqrt(2 * np.pi) * VP_K) *
                    np.exp(VP_DELTA * (4 * a**2 + b**2)) *
                    (np.exp(1j * VP_W0 * a) - np.exp(-VP_K**2 / 2)))
    return gabor_kernels


if __name__ == '__main__':
    rospy.init_node('road_preprocessor', anonymous=False)

    # CUDA_THREADS should be a power of 2
    CUDA_THREADS = rospy.get_param('~cuda_threads', 1024)
    USE_DEPRECATED_VDISP_LINE_RANSAC_FITTER = rospy.get_param(
        '~use_deprecated_ransac_line_fitter', False)

    # The following publications are for visualization only.
    PUBLISH_UDISPARITY_ROAD_FILTER = rospy.get_param(
        '~publish_udisparity_road_filter', False)
    PUBLISH_VDISPARITY_WITH_FITTED_LINE = rospy.get_param(
        '~publish_vdisparity_with_fitted_line', False)
    PUBLISH_LINE_FITTED_ROAD = rospy.get_param(
        '~publish_line_fitted_road', False)
    PUBLISH_CLOUD_COLORING = rospy.get_param('~publish_cloud_coloring', False)

    # Road Line Fit Parameters
    # Note: Disparity values are uint16.
    MAX_DISPARITY = rospy.get_param('~max_disparity', 16383)
    HISTOGRAM_BINS = rospy.get_param('~histogram_bins', 256)
    BIN_SIZE = (MAX_DISPARITY + 1) / HISTOGRAM_BINS
    FLATNESS_THRESHOLD = rospy.get_param('~flatness_threshold', 2)
    FIT_VDISP_LINE_RANSAC_TRIES_PER_THREAD = rospy.get_param(
        '~fit_vdisp_line_ransac_tries_per_thread', 100)
    FIT_VDISP_LINE_RANSAC_EPSILON = rospy.get_param(
        '~fit_vdisp_line_ransac_epsilon', 2)
    ROAD_LINE_FIT_ALPHA = rospy.get_param('~road_line_fit_alpha', 0.20)

    # Vanishing Point Detection Parameters
    VP_N = 4  # Implementation Specific
    VP_CANDIDATES_BOX_HEIGHT = 40
    VP_LAMBDA = 4 * np.sqrt(2)
    VP_KERNEL_SIZE = int(10 * VP_LAMBDA / np.pi) + 1  # Must be odd
    VP_W0 = 2 * np.pi / VP_LAMBDA
    VP_K = np.pi / 2
    VP_DELTA = -VP_W0 ** 2 / (VP_K ** 2 * 8)

    if USE_DEPRECATED_VDISP_LINE_RANSAC_FITTER:
        # Overrides get_ransac_fitted_vdisp_line
        from deprecated_vdisp_line_ransac_fitter import *
    else:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        f = open(CUDA_VDISP_LINE_RANSAC_FITTER_FILENAME, 'r')
        mod = SourceModule(Template(f.read()).substitute(
            threads=CUDA_THREADS,
            tries_per_thread=FIT_VDISP_LINE_RANSAC_TRIES_PER_THREAD,
            ransac_epsilon=FIT_VDISP_LINE_RANSAC_EPSILON), no_extern_c=True)
        f.close()

        initKernels = mod.get_function('initKernels')
        getVdispLine = mod.get_function('getVdispLine')
        initKernels(np.int32(time.time()), block=(CUDA_THREADS, 1, 1))

    cv_bridge = CvBridge()
    udisp_road_filter_image_pub = (
        rospy.Publisher('/camera/udisp_road_filter/image_rect',
                        Image, queue_size=1)
        if PUBLISH_UDISPARITY_ROAD_FILTER else None)
    vdisp_with_fitted_line_image_pub = (
        rospy.Publisher('/camera/vdisp_with_fitted_line/image_rect',
                        Image, queue_size=1)
        if PUBLISH_VDISPARITY_WITH_FITTED_LINE else None)
    line_fitted_road_image_pub = (
        rospy.Publisher('/camera/line_fitted_road/image_rect',
                        Image, queue_size=1)
        if PUBLISH_LINE_FITTED_ROAD else None)
    cloud_coloring_image_pub = (
        rospy.Publisher('/camera/cloud_coloring/image_rect',
                        Image, queue_size=1)
        if PUBLISH_CLOUD_COLORING else None)
    # TODO: Publish roadLinePub and vanishingPointPub.

    camera_image_sub = message_filters.Subscriber('/camera/left/image_rect',
                                                  Image)
    disp_image_sub = message_filters.Subscriber(
        '/camera/disp/image_rect', Image)
    ts = message_filters.TimeSynchronizer(
        [camera_image_sub, disp_image_sub], 1)
    ts.registerCallback(preprocess_road_callback,
                        cv_bridge,
                        None,  # get_gabor_filter_kernels(),
                        udisp_road_filter_image_pub,
                        vdisp_with_fitted_line_image_pub,
                        line_fitted_road_image_pub,
                        cloud_coloring_image_pub)

    rospy.spin()
