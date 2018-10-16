#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

HISTOGRAM_BINS = 1024
BIN_SIZE = 65536 / HISTOGRAM_BINS
FLATNESS_THRESHOLD = 2
bridge = CvBridge()

def roadCallback(cameraImageMsg, dispImageMsg, VdispImagePub, roadImagePub):
    dispImage = bridge.imgmsg_to_cv2(
        dispImageMsg, desired_encoding='passthrough')
    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough')
        
    UdispImage = np.apply_along_axis(
        lambda c: np.histogram(c, bins=HISTOGRAM_BINS, range=(0, 65535))[0], 
        0, 
        dispImage)
        
    rows, cols = dispImage.shape
    filterImage = UdispImage[
        dispImage / BIN_SIZE, np.mgrid[0:rows, 0:cols][1]] < FLATNESS_THRESHOLD
    
    VdispImage = 0 != np.apply_along_axis(
        lambda r: np.histogram(r, bins=HISTOGRAM_BINS, range=(0, 65535))[0], 
        1, 
        dispImage * filterImage)
    
    VdispImage = VdispImage.astype('uint16') * 65535
    filterImage = filterImage.astype('uint16') * 65535
    
    VdispImagePub.publish(bridge.cv2_to_imgmsg(
        VdispImage.astype('uint16'), encoding='16UC1'))
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
