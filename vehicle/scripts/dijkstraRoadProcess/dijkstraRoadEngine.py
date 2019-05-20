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

# Display Switches for Cost Matrixs
ShowGradientsCostMatrix = False
ShowOrientationLinkCostMatrix = False
ShowFlatnessCostMatrix = True
ShowDisparityFeatureCostMatrix = False
ShowGradientDirectionCostMatrix = False

DISPLAY_COST_TIMING = True
DISPLAY_DIJK_TIMING = True
PUBLISH_DIJKSTRA_LIMITS_ROAD = True
MOCK_VP = True

IMG_NAME = "0000000020.png"
IMAGES_FOLDER_PATH = "/home/emilio/Documents/2011_09_26/"
DEPTH_FRAME = "proj_depth/groundtruth/image_02/" + IMG_NAME
CAMERA_FRAME = "image_02/data/" + IMG_NAME



def getVanishingPoint(gray_image):
    # print gray_image
    print "waiting for user input"
    cv2.imshow('grab_vp', gray_image)
    cv2.waitKey(0)
    # return mocked_vp_x, mocked_vp_y
    vanish_X = input('enter coord X of VP :')
    vanish_Y = input('ente coord Y of VP :')
    
    cv2.destroyWindow('grab_vp')
    return (vanish_X, vanish_Y)

##############
# Depreciated
#############
def getFittedLineParams():
    k = 100
    b = 50
    return k,b


def retrieveFrames():
    cameraImage = cv2.imread(IMAGES_FOLDER_PATH + CAMERA_FRAME, cv2.IMREAD_GRAYSCALE)
    disparityImage = cv2.imread(IMAGES_FOLDER_PATH + DEPTH_FRAME, cv2.IMREAD_GRAYSCALE)

    return disparityImage, cameraImage


def processDijkstraCosts(depth_frame, 
                         camera_frame,
                         vdisp_k,
                         vdisp_m,
                         vp_x, 
                         vp_y,
                         printCosts=False):
    """
    Computes 5 costs of the Dijkstra Road Detection.
    """

    # compute imagate derivatives x, y
    sobel_x, sobel_y = computeSobelDerivatives(camera_frame)
    
    # GRADIENT COST
    gradientsCostMatrix = computeGradientCost(gx=sobel_x, gy=sobel_y)  

    # LINK COST
    # OrientationLinkCostMatrix = computeOrientationLinkCost()  
    OrientationLinkCostMatrix = sobel_x
    # print("link cost :", computeOrientationLinkCost(220,220,221,221,sobel_x,sobel_y))

    # FLATNESS COST
    flatnessCostMatrix = computeFlatnessCost(depth_frame, vdisp_m, vdisp_k)                

    # DISPARITY COST
    DisparityFeatureCostMatrix = computeDisparityFeatureCost(depth_frame)  

    # GRADIENT DIRECTION COST
    GradientDirectionCostMatrix = computeGradientDirectionCost(depth_frame, gx=sobel_x, gy=sobel_y, vp_x=vp_x, vp_y=vp_y) 

    rospy.logdebug("Summing up total weighted cost ...")
    # Addup all weights
    GRADIENT_COST_W = 0.2
    LINK_COST_W = 0.2
    FLATNESS_COST_W = 0.2
    DISPARITY_COST_W = 0.2
    GRADIENT_DIR_W  = 0.2

    weightedSumMatrix = np.asarray(GRADIENT_COST_W*gradientsCostMatrix \
        + FLATNESS_COST_W*flatnessCostMatrix \
        + DISPARITY_COST_W*DisparityFeatureCostMatrix \
        + GRADIENT_DIR_W*GradientDirectionCostMatrix,dtype=np.uint8) 
    
    rospy.logdebug("...done")

    # cv2.imshow('Weighted Sum', weightedSumMatrix)

    if (printCosts):
        # Show the costs
        displayCostImages(
            gradientsCostMatrix,
            OrientationLinkCostMatrix,
            flatnessCostMatrix,
            DisparityFeatureCostMatrix,
            GradientDirectionCostMatrix
        )

    return weightedSumMatrix

def displayCostImages(
    GradientsCost,
    OrientationLinkCost,
    FlatnessCost,
    DisparityFeatureCost,
    GradientDirectionCost):
    """
    Helper Function to display the 5 Matrix costs
    """
    # Gradient Cost
    if (ShowGradientsCostMatrix):
        cv2.imshow('Gradients Cost Matrix', GradientsCost)

    # Orientation Link Cost
    if (ShowOrientationLinkCostMatrix):
        cv2.imshow('Orientation Link Cost Matrix', OrientationLinkCost)

    # Flatness Cost Matrix
    if (ShowFlatnessCostMatrix):
        cv2.imshow('Flatness Cost Matrix', FlatnessCost)

    # Disparity Cost Matrix
    if (ShowDisparityFeatureCostMatrix):
        cv2.imshow('Disparity Feature Cost Matrix', DisparityFeatureCost)

    # Gradient Directoin Cost Matrix
    if (ShowGradientDirectionCostMatrix):
        cv2.imshow('Gradient Direction Cost Matrix', GradientDirectionCost)

def isValidNeighboor(h,w,y,x):
    return (y > 0 and y < h and x > 0 and x < w)


def grab_vp(event,x,y,flags,param):
    global mocked_vp_x, mocked_vp_y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print "x = ", x, "y = ", y
        mocked_vp_x, mocked_vp_y = x,y

def dijkstraRoadDetectionCallback(color_image_msg,
                                  disp_image_msg,
                                  vdisp_line_msg,
                                  cv_bridge,
                                  dijkstra_line_limits_road_image_pub,
                                  ):
    """
    This callback will compute dijkstra road model detection.
    """
    # ------------------------ DATA GRAB ---------------------- #
    # retrieve images from ROS msgs to cv mats
    disp_image = cv_bridge.imgmsg_to_cv2(
        disp_image_msg, desired_encoding='passthrough')
    color_image = cv_bridge.imgmsg_to_cv2(
        color_image_msg, desired_encoding='passthrough')
    vdisp_k = vdisp_line_msg.b
    vdisp_m = vdisp_line_msg.m 

    # disp_normalized = np.zeros(shape=color_image.shape, type=np.uint8)
    # np.interp(disp_normalized, (disp_image.min(), disp_image.max()), (0, 255))

    disp_scaled = cv2.normalize(disp_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

    # cv2.imshow('color frame', color_image)
    # cv2.imshow('disparity frame', disp_scaled)
    # print "Max at disparity ", disp_image.max()
    # cv2.waitKey(50)

    # ------------------------ COSTS PROCESSING ---------------------- #
    # converting to grayscale
    gray_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)

    # img dims
    frame_height, frame_width = disp_scaled.shape

    # retriving vanishing point coordenates
    vp_x, vp_y = getVanishingPoint(gray_image)

    # trimming working frames based on vanishing point
    t_disp_frame = disp_scaled[vp_y:,:]
    t_camera_frame = gray_image[vp_y:,:]
    
    t_beginCosts = time.time()
    # Weighted sum of costs.
    # Note than Max should not be greather than 255.
    rospy.logdebug("COMPUTING DIJKSTRA COSTS ... ")
    weightedSumMatrix = processDijkstraCosts( 
                        t_disp_frame, 
                        t_camera_frame,
                        vdisp_k=vdisp_k,
                        vdisp_m=vdisp_m,
                        vp_x=vp_x,
                        vp_y=vp_y,
                        printCosts=False)

    # make sure no overflows on 8bit image
    assert weightedSumMatrix.max() <= 255 and weightedSumMatrix.min() >= 0
    rospy.logdebug("\t...done")

    # display time to compute costs
    if DISPLAY_COST_TIMING:
        print "Time to compute dijkstra costs: ", time.time() - t_beginCosts 
    
    print "Matrix Max, Min : ", weightedSumMatrix.max(), weightedSumMatrix.min()
    
    # cv2.waitKey(100)
    # rospy.sleep(1)

    # ------------------------ DIJKSTRA UTILITY ---------------------- #
    t_begin_dij = time.time()

    visualizationFrame =  cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)
    weightedSumMatrix_left = weightedSumMatrix[:,0:vp_x+1]
    weightedSumMatrix_right = weightedSumMatrix[:,vp_x-1:frame_width] # VP for right is in the coord (0,1)
    
    shortest_path_left = dijU.dynamicDijkstraRoadDetection(weightedSumMatrix_left, vp_x)
    shortest_path_right = dijU.dynamicDijkstraRoadDetection(weightedSumMatrix_right, 1)
    
    if DISPLAY_DIJK_TIMING:
        print "Time to compute dijstra path: ", time.time() - t_begin_dij

    drivableMatrix = np.zeros((frame_height-vp_y, frame_width,3), np.uint8)
    # draw the path
    print "Nodes from left : {} \nnodes from right : {}".format(len(shortest_path_left), len(shortest_path_right))
    
    rospy.logdebug("drawing paths")
    for p in shortest_path_left:
         cv2.circle(visualizationFrame,(p[1], p[0]+vp_y), 1, (0,255,0), -1)
    
    for p in shortest_path_right:
         cv2.circle(visualizationFrame,(p[1]+vp_x, p[0]+vp_y), 1, (0,0,255), -1)

    rospy.logdebug("done")

    rospy.logdebug("imshowing vis")
    cv2.imshow("preview", visualizationFrame)
    cv2.imshow("costs sum", weightedSumMatrix)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rospy.logdebug("done")

    
    # cv2.destroyAllWindows()

    # cv2.imshow('Camera Image', camera_frame)
    # cv2.imshow('Disparity Image', disparityImage)
    # a = raw_input('enter anything to continue')
    

#################################################################################
#                                   MAIN                                        #
#################################################################################
if __name__ == "__main__":

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

    
