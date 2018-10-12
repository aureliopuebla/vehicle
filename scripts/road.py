#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np

HISTOGRAM_BINS = 20
bridge = CvBridge()

def dispCallback(msg, pub):
    dispImage = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    UdispImage = np.apply_along_axis(
        lambda e: np.histogram(e, bins=HISTOGRAM_BINS, range=(0, 255))[0], 
        0, 
        dispImage)
    VdispImage = np.apply_along_axis(
        lambda e: np.histogram(e, bins=HISTOGRAM_BINS, range=(0, 255))[0], 
        1, 
        dispImage)
    pub.publish(bridge.cv2_to_imgmsg(
        UdispImage.astype('uint16'), encoding='passthrough'))
    
def listener():
    rospy.init_node('road', anonymous=False)
    pub = rospy.Publisher('/camera/road', Image, queue_size=1)
    rospy.Subscriber("/camera/disp", Image, dispCallback, pub)
    rospy.spin()

if __name__ == '__main__':
    listener()
