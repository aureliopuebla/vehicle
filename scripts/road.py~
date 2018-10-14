#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

HISTOGRAM_BINS = 50
FLATNESS_THRESHOLD = 2
bridge = CvBridge()

def roadCallback(cameraImageMsg, dispImageMsg, roadImagePub):
    dispImage = bridge.imgmsg_to_cv2(
        dispImageMsg, desired_encoding='passthrough')
    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough')
        
    UdispImage = np.apply_along_axis(
        lambda e: np.histogram(e, bins=HISTOGRAM_BINS, range=(0, 255))[0], 
        0, 
        dispImage)
    roadImage = np.apply_along_axis(
        lambda e: e, 
        0, 
        dispImage)
    
    roadImagePub.publish(bridge.cv2_to_imgmsg(
        cameraImage, encoding='bgr8'))
    
def listener():
    rospy.init_node('road', anonymous=False)
    
    roadImagePub = rospy.Publisher('/camera/road', Image, queue_size=1)
    
    cameraImageSub = message_filters.Subscriber('/camera/image', Image)
    dispImageSub = message_filters.Subscriber('/camera/disp', Image)
    ts = message_filters.TimeSynchronizer([cameraImageSub, dispImageSub], 10)
    ts.registerCallback(roadCallback, roadImagePub)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
