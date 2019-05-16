#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vehicle.msg import VdispLine
from vehicle.msg import VanishingPoint

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2

CUDA_VANISHING_POINT_DETECTOR_FILENAME = 'vanishing_point_detector.cu'


def get_gabor_filter_kernel(theta, frequency, sigma_x, sigma_y, size):
    """Returns both the imaginary and real parts of a Gabor kernel.

    A Gabor kernel is a Gaussian kernel modulated by a complex harmonic
    function whose real and imaginary parts consist of a cosine and sine
    functions respectively.
    Harmonic function consists of an imaginary sine function and a real
    cosine function.

    Args:
      frequency : Frequency of the harmonic function in pixels.
      theta: Orientation in radians. If 0 is the x-direction.
      sigma_x: Standard deviation in x direction.
      sigma_y: Same as sigma_x but in the y direction. This applies to the
          kernel before rotation.
      size: Kernel size which applies to both dimensions.
    """
    y, x = np.mgrid[-int(size / 2):int(size / 2) + 1,
                    -int(size / 2):int(size / 2) + 1]

    rot_x = x * np.cos(theta) + y * np.sin(theta)
    rot_y = -x * np.sin(theta) + y * np.cos(theta)

    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(
        -0.5 * (rot_x ** 2 / sigma_x ** 2 + rot_y ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.exp(1j * (2 * np.pi * frequency * rot_x))

    return g.real.astype(np.float32), g.imag.astype(np.float32)


def get_gabor_filter_kernels(thetas, frequencies):
    """Returns an array of Gabor filter kernels where position [i][j][k] means
       the kernel of ith theta, jth frequency, kth complex part (where 0 = real,
       1 = imag)."""
    return [[get_gabor_filter_kernel(theta,
                                     frequency,
                                     sigma_x=GABOR_FILTER_SIGMA_X,
                                     sigma_y=GABOR_FILTER_SIGMA_Y,
                                     size=GABOR_FILTER_KERNEL_SIZE)
             for frequency in frequencies]
            for theta in thetas]


def apply_gabor_kernels(grey_image, gabor_kernels):
    original_rows, original_cols = grey_image.shape
    response_rows = original_rows - GABOR_FILTER_KERNEL_SIZE + 1
    response_cols = original_cols - GABOR_FILTER_KERNEL_SIZE + 1
    grey_image_gpu = cuda.In(grey_image)

    energies = np.empty((response_rows, response_cols, THETAS_N),
                        dtype=np.float32)
    for i in range(THETAS_N):
        real_kernel = gabor_kernels[i][0][0]
        imag_kernel = gabor_kernels[i][0][1]

        real_kernel_gpu = cuda.mem_alloc(real_kernel.nbytes)
        imag_kernel_gpu = cuda.mem_alloc(imag_kernel.nbytes)
        response_gpu = cuda.mem_alloc(
            np.float32().itemsize * (response_rows) * (response_cols))

        cuda.memcpy_htod(real_kernel_gpu, real_kernel)
        cuda.memcpy_htod(imag_kernel_gpu, imag_kernel)
        complexFilter2D(
            grey_image_gpu, np.int32(original_rows), np.int32(original_cols),
            response_gpu, real_kernel_gpu, imag_kernel_gpu,
            np.int32(GABOR_FILTER_KERNEL_SIZE),
            block=(16, 16, 1), grid=(original_cols / 16, original_rows / 16))
        response = np.empty((response_rows, response_cols), np.float32)
        cuda.memcpy_dtoh(response, response_gpu)

        energies[:, :, i] = response
    return energies


def get_vanishing_point_callback(color_image_msg,
                                 vdisp_line,
                                 cv_bridge,
                                 cuda_context,
                                 gabor_kernels,
                                 vanishing_point_pub,
                                 gabor_filtered_images_pubs=None):
    """Publishes the vanishing_point by processing the camera's grey_image,
    exploiting the previously obtained vdisp_line.

    Args:
      color_image_msg: A ROS Message containing the road color Image.
          (TODO: Change this to the grey_image_msg once in recordings)
      vdisp_line: A ROS Message containing the vdisparity fitted line.
      cv_bridge: The CV bridge instance used to convert CV images and ROS
          Messages.
      cuda_context: The CUDA context to be used to execute kernels.
      vanishing_point_pub: The ROS Publisher for the vanishing_point.
      gabor_filtered_images_pubs: If set, it's an array of ROS Publishers
          where each Publisher matches a theta of the applied Gabor kernels.
    """
    color_image = cv_bridge.imgmsg_to_cv2(
        color_image_msg, desired_encoding='passthrough')
    grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    grey_image = grey_image[
        int(vdisp_line.b) - VP_HORIZON_CANDIDATES_MARGIN:, :]

    cuda_context.push()
    original_rows, original_cols = grey_image.shape
    response_rows = original_rows - GABOR_FILTER_KERNEL_SIZE + 1
    response_cols = original_cols - GABOR_FILTER_KERNEL_SIZE + 1

    energies = apply_gabor_kernels(grey_image, gabor_kernels)
    energies_gpu = cuda.mem_alloc(energies.nbytes)
    combined_gpu = cuda.mem_alloc(
        np.float32().itemsize * response_rows * response_cols)

    cuda.memcpy_htod(energies_gpu, energies)
    combineFilteredImages(
        energies_gpu, np.int32(response_rows), np.int32(response_cols),
        combined_gpu, block=(16, 16, 1),
        grid=(int(np.ceil(response_cols / 16)),
              int(np.ceil(response_rows / 16))))
    combined = np.empty((response_rows, response_cols), np.float32)
    cuda.memcpy_dtoh(combined, combined_gpu)

    vanishing_point_pub.publish(  # TODO: Publish real VP
        VanishingPoint(header=color_image_msg.header, row=0, col=0))

    if gabor_filtered_images_pubs is not None:
        for i in range(THETAS_N):
            kernel = gabor_kernels[i][0][0].copy()
            kernel -= kernel.min()
            kernel /= kernel.max()
            kernel *= 255
            energy = energies[:, :, i]
            energy -= energy.min()
            energy /= energy.max()
            energy *= 255
            combined -= combined.min()
            combined /= combined.max()
            combined *= 255
            gabor_filtered_images_pubs[i].publish(cv_bridge.cv2_to_imgmsg(
                #kernel.astype('uint8'),
                #combined.astype('uint8'),
                energy.astype('uint8'),
                encoding='8UC1'))


if __name__ == '__main__':
    rospy.init_node('vanishing_point_detector', anonymous=False)

    # The following publications are for debugging and visualization only as
    # they severely slow down node execution.
    PUBLISH_GABOR_FILTERED_IMAGES = rospy.get_param(
        '~publish_gabor_filtered_images', False)

    # Vanishing Point Detection Parameters
    VP_HORIZON_CANDIDATES_MARGIN = rospy.get_param(
        '~vp_horizon_candidates_margin', 20)
    THETAS_N = 4  # Implementation Specific
    GABOR_FILTER_THETAS = [np.pi / THETAS_N * i for i in range(THETAS_N)]
    GABOR_FILTER_KERNEL_SIZE = rospy.get_param(
        '~gabor_filter_kernel_size', 21)  # Should be odd.
    assert GABOR_FILTER_KERNEL_SIZE % 2 == 1
    GABOR_FILTER_SIGMA_X = rospy.get_param(
        '~gabor_filter_sigma_x', 4)
    GABOR_FILTER_SIGMA_Y = rospy.get_param(
        '~gabor_filter_sigma_y', 2)
    # Recommendation: Values between 0.1 and 0.5
    GABOR_FILTER_FREQUENCIES = rospy.get_param(
        '~gabor_filter_frequencies', [0.25, 0.3, 0.35])

    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()
    with open(CUDA_VANISHING_POINT_DETECTOR_FILENAME, 'r') as f:
        mod = SourceModule(f.read(), no_extern_c=True)
    complexFilter2D = mod.get_function('complexFilter2D')
    combineFilteredImages = mod.get_function('combineFilteredImages')

    cv_bridge = CvBridge()
    vanishing_point_pub = rospy.Publisher(
        'vanishing_point', VanishingPoint, queue_size=1)
    gabor_filtered_images_pubs = (
        [rospy.Publisher('gabor_%d_filtered_image' % (180 / THETAS_N * i),
                         Image,
                         queue_size=1)
         for i in range(THETAS_N)]
        if PUBLISH_GABOR_FILTERED_IMAGES else None)

    color_image_sub = message_filters.Subscriber(
        '/multisense/left/image_rect_color', Image)
    vdisp_image_sub = message_filters.Subscriber(
        '/vdisp_line_fitter/vdisp_line', VdispLine)
    ts = message_filters.TimeSynchronizer(
        [color_image_sub, vdisp_image_sub], queue_size=5)
    ts.registerCallback(get_vanishing_point_callback,
                        cv_bridge,
                        cuda_context,
                        get_gabor_filter_kernels(
                            GABOR_FILTER_THETAS, GABOR_FILTER_FREQUENCIES),
                        vanishing_point_pub,
                        gabor_filtered_images_pubs)

    rospy.spin()

