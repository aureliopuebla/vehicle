#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import deprecated_vdisp_line_fitter_code as deprecated
from vehicle.msg import VdispLine

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import string
import time

CUDA_VDISP_LINE_FITTER_FILENAME = 'vdisp_line_fitter.cu'


def get_udisp_threshold_filter(disp_image, udisp_image):
    """Uses the udisp_image to return a boolean filter for disp_image where
       'True' is assigned to a given (r,c) coordinate iff its corresponding
       udisp value is below a given FLATNESS_THRESHOLD. Such udisp value is
       given by the frequency value that corresponds to the disparity value
       (r,c) in the histogram of column 'c'."""
    rows, cols = disp_image.shape
    return udisp_image[np.minimum(HISTOGRAM_BINS - 1, disp_image / BIN_SIZE),
                       np.mgrid[0:rows, 0:cols][1]] < FLATNESS_THRESHOLD


def get_road_line_fit_filter(disp_image, m, b):
    """Returns a boolean filter for the original road image of the road values
       that are close to the fitted road line."""
    rows, _ = disp_image.shape
    road_row_values = np.fromfunction(
        np.vectorize(lambda r, _: float(r - b) / m), (rows, 1))
    return np.abs(
        disp_image - road_row_values) <= ROAD_LINE_FIT_ALPHA * road_row_values


def get_ransac_fitted_vdisp_line(
        vdisp_image_gpu, rows, bins, vdisp_cum_sum_array_gpu):
    """Gets the vdisp_line using RANSAC running on CUDA."""
    m = np.empty(1, dtype=np.float32)
    b = np.empty(1, dtype=np.float32)
    getVdispLine(
        vdisp_image_gpu, np.int32(rows), np.int32(bins),
        vdisp_cum_sum_array_gpu, cuda.Out(m), cuda.Out(b),
        block=(CUDA_THREADS, 1, 1))
    return m[0] / BIN_SIZE, b[0]


def get_vdisp_line_callback(color_image_msg,
                            disp_image_msg,
                            cv_bridge,
                            cuda_context,
                            vdisp_line_pub,
                            udisp_threshold_filter_image_pub=None,
                            vdisp_with_fitted_line_image_pub=None,
                            line_fitted_road_image_pub=None,
                            cloud_coloring_image_pub=None):
    """Publishes the vdisp_line by processing the disp_image.

    First, from the disp_image, a set of histograms in the vertical axis, known
    as the udisp_image, is obtained. This udisp_image is used to filter the
    disp_image, keeping only the pixel values whose corresponding udisp value is
    below a certain threshold. Then, a histogram of such kept disp_image values
    is obtained in the horizontal direction; this is the vdisp_image. Finally,
    RANSAC is used to fit a line through the vdisp_image getting what's known as
    the vdisp_line which relates rows to disparities in the road plane.

    Args:
      color_image_msg: A ROS Message containing the road color Image.
      disp_image_msg: A ROS Message containing the corresponding road disparity
          Image.
      cv_bridge: The CV bridge instance used to convert CV images and ROS
          Messages.
      cuda_context: The CUDA context to be used to execute kernels.
      vdisp_line_pub: The ROS Publisher for the vdisparity fitted line.
      udisp_threshold_filter_image_pub: If set, it's the ROS Publisher that
          will contain the udisparity filter for visualization.
      vdisp_with_fitted_line_image_pub: If set, it's the ROS Publisher that will
          contain the vdisparity with the RANSAC fitted line for visualization.
      line_fitted_road_image_pub: If set, it's the ROS Publisher that will
          contain the Line Fitted Road Image with the horizon line for
          visualization.
      cloud_coloring_image_pub: If set, it's the ROS Publisher that will contain
        the corresponding colors for the point cloud based on a color code where
        YELLOW are obstacles and WHITE is the vdisp filtered road.
    """
    disp_image = cv_bridge.imgmsg_to_cv2(
        disp_image_msg, desired_encoding='passthrough')
    color_image = cv_bridge.imgmsg_to_cv2(
        color_image_msg, desired_encoding='passthrough')

    if USE_DEPRECATED_CODE:
        udisp_image = np.apply_along_axis(
            deprecated.get_histogram, 0, disp_image)
        udisp_filter = get_udisp_threshold_filter(disp_image, udisp_image)
        vdisp_image = np.apply_along_axis(
            deprecated.get_histogram, 1, disp_image * udisp_filter)
        m, b = deprecated.get_ransac_fitted_vdisp_line(vdisp_image)

    else:
        # Note: Optimized for disp_image of type uint16.
        cuda_context.push()
        rows, cols = disp_image.shape

        disp_image_gpu = cuda.mem_alloc(disp_image.nbytes)
        udisp_image_gpu = cuda.mem_alloc(
            np.uint16().itemsize * HISTOGRAM_BINS * cols)
        vdisp_image_gpu = cuda.mem_alloc(
            np.uint16().itemsize * rows * HISTOGRAM_BINS)
        vdisp_cum_sum_array_gpu = cuda.mem_alloc(
            np.int32().itemsize * rows * HISTOGRAM_BINS)

        cuda.memcpy_htod(disp_image_gpu, disp_image)
        getUDisparity(
            disp_image_gpu, np.int32(rows), np.int32(cols),
            udisp_image_gpu, np.int32(HISTOGRAM_BINS), np.int32(BIN_SIZE),
            block=(cols, 1, 1))
        getVDisparity(
            disp_image_gpu, np.int32(rows), np.int32(cols),
            udisp_image_gpu, np.int32(FLATNESS_THRESHOLD),
            vdisp_image_gpu, np.int32(HISTOGRAM_BINS), np.int32(BIN_SIZE),
            block=(rows, 1, 1))
        getCumSumArray(
            vdisp_image_gpu, vdisp_cum_sum_array_gpu, np.int32(
                rows * HISTOGRAM_BINS),
            block=(CUDA_THREADS, 1, 1))
        m, b = get_ransac_fitted_vdisp_line(
            vdisp_image_gpu, rows, HISTOGRAM_BINS, vdisp_cum_sum_array_gpu)

    vdisp_line_pub.publish(VdispLine(header=disp_image_msg.header, m=m, b=b))

    if udisp_threshold_filter_image_pub is not None:
        if not USE_DEPRECATED_CODE:
            udisp_image = np.empty((HISTOGRAM_BINS, cols), np.uint16)
            cuda.memcpy_dtoh(udisp_image, udisp_image_gpu)
            udisp_filter = get_udisp_threshold_filter(disp_image, udisp_image)
        # Convert Binary Image to uint8.
        udisp_threshold_filter_image_pub.publish(cv_bridge.cv2_to_imgmsg(
            udisp_filter.astype('uint8') * 255, encoding='8UC1'))

    if vdisp_with_fitted_line_image_pub is not None:
        if not USE_DEPRECATED_CODE:
            vdisp_image = np.empty((rows, HISTOGRAM_BINS), np.uint16)
            cuda.memcpy_dtoh(vdisp_image, vdisp_image_gpu)
        # Show elements with values > 0.
        vdisp_image = vdisp_image.astype('uint8') * 255
        vdisp_with_fitted_line = cv2.cvtColor(vdisp_image, cv2.COLOR_GRAY2RGB)
        _, cols = vdisp_image.shape
        cv2.line(vdisp_with_fitted_line,
                 (0, int(b)),
                 (cols - 1, int((cols - 1) * (m * BIN_SIZE) + b)),
                 (0, 0, 255),
                 2)
        vdisp_with_fitted_line_image_pub.publish(
            cv_bridge.cv2_to_imgmsg(vdisp_with_fitted_line, encoding='bgr8'))

    if line_fitted_road_image_pub is not None:
        line_fitted_road_filter = get_road_line_fit_filter(disp_image, m, b)
        line_fitted_road = (
            color_image * line_fitted_road_filter[:, :, np.newaxis])
        cv2.line(line_fitted_road,
                 pt1=(0, int(b)),
                 pt2=(color_image.shape[1] - 1, int(b)),
                 color=(0, 0, 255),
                 thickness=2)
        line_fitted_road_image_pub.publish(
            cv_bridge.cv2_to_imgmsg(line_fitted_road, encoding='bgr8'))

    if cloud_coloring_image_pub is not None:
        line_fitted_road_filter = get_road_line_fit_filter(disp_image, m, b)
        cloud_coloring = 255 * np.ones(color_image.shape, np.uint8)
        cloud_coloring[:, :, 0:1] *= line_fitted_road_filter[:, :, np.newaxis]
        cloud_coloring_image_msg = cv_bridge.cv2_to_imgmsg(
            cloud_coloring, encoding='bgr8')
        cloud_coloring_image_msg.header = disp_image_msg.header
        cloud_coloring_image_pub.publish(cloud_coloring_image_msg)


if __name__ == '__main__':
    rospy.init_node('vdisp_line_fitter', anonymous=False)

    # CUDA_THREADS should be a power of 2.
    CUDA_THREADS = rospy.get_param('~cuda_threads', 1024)
    USE_DEPRECATED_CODE = rospy.get_param('~use_deprecated_code', False)

    # The following publications are for debugging and visualization only as
    # they severely slow down node execution.
    PUBLISH_UDISP_THRESHOLD_FILTER = rospy.get_param(
        '~publish_udisp_threshold_filter', False)
    PUBLISH_VDISP_WITH_FITTED_LINE = rospy.get_param(
        '~publish_vdisp_with_fitted_line', False)
    PUBLISH_VDISP_LINE_FITTED_ROAD = rospy.get_param(
        '~publish_vdisp_line_fitted_road', False)
    PUBLISH_CLOUD_COLORING = rospy.get_param('~publish_cloud_coloring', False)

    # Vdisp Line Fitting Parameters
    # Note: Disparity values are uint16.
    MAX_DISPARITY = rospy.get_param('~max_disparity', 4096)
    HISTOGRAM_BINS = rospy.get_param('~histogram_bins', 256)
    BIN_SIZE = (MAX_DISPARITY + 1) / HISTOGRAM_BINS
    FLATNESS_THRESHOLD = rospy.get_param('~flatness_threshold', 16)
    FIT_VDISP_LINE_RANSAC_TRIES_PER_THREAD = rospy.get_param(
        '~fit_vdisp_line_ransac_tries_per_thread', 20)
    FIT_VDISP_LINE_RANSAC_EPSILON = rospy.get_param(
        '~fit_vdisp_line_ransac_epsilon', 2)
    FIT_VDISP_LINE_RANSAC_EPSILON_DECAY = rospy.get_param(
        '~fit_vdisp_line_ransac_epsilon_decay', 0.25)
    ROAD_LINE_FIT_ALPHA = rospy.get_param('~road_line_fit_alpha', 0.15)

    if not USE_DEPRECATED_CODE:
        cuda.init()
        cuda_device = cuda.Device(0)
        cuda_context = cuda_device.make_context()
        with open(CUDA_VDISP_LINE_FITTER_FILENAME, 'r') as f:
            mod = SourceModule(
                string.Template(
                    f.read()).substitute(
                    threads=CUDA_THREADS,
                    tries_per_thread=FIT_VDISP_LINE_RANSAC_TRIES_PER_THREAD,
                    ransac_epsilon=FIT_VDISP_LINE_RANSAC_EPSILON,
                    ransac_epsilon_decay=FIT_VDISP_LINE_RANSAC_EPSILON_DECAY),
                no_extern_c=True)
        initRandomStates = mod.get_function('initRandomStates')
        getUDisparity = mod.get_function('getUDisparity')
        getVDisparity = mod.get_function('getVDisparity')
        getCumSumArray = mod.get_function('getCumSumArray')
        getVdispLine = mod.get_function('getVdispLine')
        initRandomStates(np.int32(time.time()), block=(CUDA_THREADS, 1, 1))

    cv_bridge = CvBridge()
    vdisp_line_pub = rospy.Publisher('vdisp_line', VdispLine, queue_size=1)
    udisp_threshold_filter_image_pub = (
        rospy.Publisher('udisp_threshold_filter_image', Image, queue_size=1)
        if PUBLISH_UDISP_THRESHOLD_FILTER else None)
    vdisp_with_fitted_line_image_pub = (
        rospy.Publisher('vdisp_with_fitted_line_image', Image, queue_size=1)
        if PUBLISH_VDISP_WITH_FITTED_LINE else None)
    line_fitted_road_image_pub = (
        rospy.Publisher('vdisp_line_fitted_road_image', Image, queue_size=1)
        if PUBLISH_VDISP_LINE_FITTED_ROAD else None)
    cloud_coloring_image_pub = (
        rospy.Publisher('cloud_coloring_image', Image, queue_size=1)
        if PUBLISH_CLOUD_COLORING else None)

    color_image_sub = message_filters.Subscriber(
        '/multisense/left/image_rect_color', Image)
    disp_image_sub = message_filters.Subscriber(
        '/multisense/left/disparity', Image)
    ts = message_filters.TimeSynchronizer(
        [color_image_sub, disp_image_sub], queue_size=2)
    ts.registerCallback(get_vdisp_line_callback,
                        cv_bridge,
                        None if USE_DEPRECATED_CODE else cuda_context,
                        vdisp_line_pub,
                        udisp_threshold_filter_image_pub,
                        vdisp_with_fitted_line_image_pub,
                        line_fitted_road_image_pub,
                        cloud_coloring_image_pub)

    rospy.spin()
