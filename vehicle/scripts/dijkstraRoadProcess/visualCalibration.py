#!/usr/bin/python

# ROS imports
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vehicle.msg import VdispLine

# Opencv and numpy
import numpy as np
import math
import cv2

# utilities
import dijkstraUtility as dijU
from computeCostUtility import *
import time




if __name__ == '__main__':
    rospy.init_node('dijkstraRoadProcess', anonymous=False, log_level=rospy.DEBUG)

    # ROS subscriptions disparity and color img
    color_image_sub = message_filters.Subscriber(
        '/multisense/left/image_rect_color', Image)
    disp_image_sub = message_filters.Subscriber(
        '/multisense/left/disparity', Image)

    # ROS subscription vdisp_line 
    vdisp_line_sub = message_filters.Subscriber(
        '/vdisp_line_fitter/vdisp_line', VdispLine, queue_size=1)
    
    # publisher node for dijstra road limits
    dijkstra_line_limits_road_image_pub = (
    rospy.Publisher('dijkstra_limits_road_image', Image, queue_size=1)
    if PUBLISH_DIJKSTRA_LIMITS_ROAD else None)
    
    # cv_bridge utility
    cv_bridge = CvBridge()

    # # Mocking VP
    # cv2.namedWindow('grab_vp')
    # cv2.setMouseCallback('grab_vp',grab_vp)

    # message filters for syncronization
    ts = message_filters.TimeSynchronizer([color_image_sub, disp_image_sub, vdisp_line_sub], 1)
    ts.registerCallback(dijkstraRoadDetectionCallback,
                        cv_bridge,
                        dijkstra_line_limits_road_image_pub
                        )

    rospy.spin()
