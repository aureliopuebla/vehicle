#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vehicle.msg import VdispLine
from vehicle.msg import VanishingPoint

import numpy as np
import cv2


def get_vanishing_point_callback(color_image_msg,
                                 vdisp_line,
                                 cv_bridge,
                                 vanishing_point_pub,
                                 gabor_filtered_image_pubs=None):
    """Publishes the vanishing_point by processing the camera's grey_image,
    exploiting the previously obtained vdisp_line.

    Args:
      color_image_msg: A ROS Message containing the road color Image.
          (TODO: Change this to the grey_image_msg once in recordings)
      vdisp_line: A ROS Message containing the vdisparity fitted line.
      cv_bridge: The CV bridge instance used to convert CV images and ROS
          Messages.
      vanishing_point_pub: The ROS Publisher for the vanishing_point.
      gabor_filtered_image_pubs: If set, it's an array of ROS Publishers
          where each Publisher matches a theta of the applied Gabor kernels.
    """
    color_image = cv_bridge.imgmsg_to_cv2(
        color_image_msg, desired_encoding='passthrough')
    grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    vanishing_point_pub.publish(
        VanishingPoint(header=color_image_msg.header, row=0, col=0))



if __name__ == '__main__':
    rospy.init_node('vanishing_point_detector', anonymous=False)

    # Vanishing Point Detection Parameters
    VP_THETAS = [np.pi / 4 * i for i in range(4)]  # Implementation Specific

    VP_LAMBDA = 4 * np.sqrt(2)
    VP_KERNEL_SIZE = int(10 * VP_LAMBDA / np.pi) + 1  # Must be odd
    VP_W0 = 2 * np.pi / VP_LAMBDA
    VP_K = np.pi / 2
    VP_DELTA = -VP_W0 ** 2 / (VP_K ** 2 * 8)

    # The following publications are for debugging and visualization only as
    # they severely slow down node execution.
    PUBLISH_GABOR_FILTERED_IMAGES = rospy.get_param(
        '~publish_gabor_filtered_images', False)

    VP_HORIZON_CANDIDATES_MARGIN = rospy.get_param(
        '~vp_horizon_candidates_margin', 20)
    GABOR_FILTER_FREQUENCIES = rospy.get_param(
        '~gabor_filter_frequencies', [1, 2, 4, 8])

    cv_bridge = CvBridge()
    vanishing_point_pub = rospy.Publisher(
        'vanishing_point', VanishingPoint, queue_size=1)

    color_image_sub = message_filters.Subscriber(
        '/multisense/left/image_rect_color', Image)
    vdisp_image_sub = message_filters.Subscriber(
        '/vdisp_line_fitter/vdisp_line', VdispLine)
    ts = message_filters.TimeSynchronizer(
        [color_image_sub, vdisp_image_sub], queue_size=5)
    ts.registerCallback(get_vanishing_point_callback,
                        cv_bridge,
                        vanishing_point_pub,
                        None)

    rospy.spin()

