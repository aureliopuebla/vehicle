#!/usr/bin/python

import numpy as np
import math
import cv2
import time

import rospy

def computeGradientDirectionCost(disparityImage, gx, gy, vp_x, vp_y):
    """
    To compute the cost of Fgd(Pv):
        Let:
            vp = VanishingPoint
            P = current point
            gradDir(P) = Gradient direction at pixel P
            vector_vp = vector linking vanishing point vp and pixel P
        THEN:
            gamaAngle = angle(vector_vp, gradDir(P))

    Hence, GradientDirectionCost:
                    |   1,   if gamaAngle <= 1 -          dist(vp,P)
                    |                           --------------------------   * Betha
                    |                            max{dist(vp, left_bottom), 
        Fgd(Pv) =   {                            dist(vp, right_bottom)}
                    |
                    |                                 
                    |   0,   otherwise.    
    """
    def getDist(x1, y1, x2, y2):
        return np.float32(math.sqrt((y2-y1)**2 + (x2-x1)**2))

    def toGrads(angle):
        return angle*180/np.pi

    def showGradients(img, step=50, line_len=30, tag='Gradients arrows'):
        tempImg = np.zeros(shape=img.shape)
        height, width = tempImg.shape
        for x in range(0, width, step):
            for y in range(0, height, step):
                x2 = int(x + line_len*math.cos(img[y, x]))
                y2 = int(y + line_len*math.sin(img[y, x]))
                cv2.line(tempImg, (x, y), (x2, y2), (255, 0, 0), 1)
        cv2.imshow(tag, tempImg)

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Entered computeGradientDirectionCost") 
    
    # ------------------------ CREATIN GRADIENT GradDir(p) ---------------------- #
    # Create matrix to allocate gradients of the img
    gradDir = np.zeros(shape=disparityImage.shape, dtype=np.float32)

    # Gradient direction is given by inverse tangent of the cocient of Y and X --> y/x
    # g(x,y) = arctan2(y,x)
    gradDir = toGrads(np.arctan2(gy, gx))
    # showGradients(gradDir,tag='normal')

    # For the purspose of the gradient in this application we need to take the smallest angle
    # between the Gradient direciton gradDir(p) and vector_vp
    gradDirInv = toGrads(np.arctan2(-gy, -gx))
    # print ("normal angle : {}, inverse angle: {}".format(gradDir[60,60], gradDirInv[60,60]))
    # showGradients(gradDirInv,tag='inverse')
    
    # ------------------------ CREATING VECTOR vector_vp ---------------------- #

    # Allocating space for vector vp
    vector_vp = np.zeros(shape=disparityImage.shape, dtype=np.float32)
    
    # computing max distance from vp to BL and BR

    height, width = disparityImage.shape
    # print height, width

    for x in range(width):
        for y in range(height):
            # This normalization need to be explained
            # vector_vp[y, x] = -(toGrads(np.arctan2((y-vp_y), (x-vp_x))) - 180)
            vector_vp[y, x] = toGrads(np.arctan2((y-vp_y), (x-vp_x)))

    # # showGradients(vector_vp*np.pi/180,tag='vp')
    # vp = vector_vp[200,200]
    # a = gradDir[200,200]
    # b = gradDirInv[200,200]
    # print ("vector vp angle :{}, gradangle : {}, invGrada : {}".format(vp,a,b))

    # ------------------------ COMPUTING GAMMA ANGLE ---------------------- #
    gamaAngleNormal =  abs(vector_vp - gradDir)
    gamaAngleInverse = abs(vector_vp - gradDirInv )
    # gamaAngle = np.minimum(gamaAngleNormal,gamaAngleInverse)
    gamaAngle = gamaAngleNormal
    
    # print (gamaAngle[200,200])
   
   
    # cv2.imshow('Gamma Angle', gamaAngle)
    # quotas = gamaAngle.copy()

    # draw lines to make sure poiting to vp
    # lines_to_vp = np.zeros(shape=disparityImage.shape)
    # line_len = 30
    # for x in range(0, width, 30):
    #     for y in range(0, height, 30):
    #         x2 = int(x + line_len*math.cos(vp_p_dir[y, x]))
    #         y2 = int(y + line_len*math.sin(vp_p_dir[y, x]))
    #         cv2.line(lines_to_vp, (x, y), (x2, y2), (255, 0, 0), 1)

    # cv2.line(lines_to_vp,(vanish_X,vanish_Y),(0,height),(255,0,0),5)
    # cv2.line(lines_to_vp,(vanish_X,vanish_Y),(width,height),(255,0,0),5)
    # cv2.imshow('were are lines pointing???', lines_to_vp)

    gradDirCosts = np.asarray(disparityImage.copy())

    # Distances from VP to corners
    DIST_VP_TO_LB = getDist(0, height, vp_x, vp_y)
    DIST_VP_TO_RB = getDist(width-1, height-1, vp_x, vp_y)
    
    # ------------------------ COMPUTE Fgd(Pv) ---------------------- #
    for x in range(width):
        for y in range(height):
            DIST_XY_TO_VP = getDist(x, y, vp_x, vp_y)
            B = 20
            localQuota = (1 - DIST_XY_TO_VP/(max(DIST_VP_TO_LB, DIST_VP_TO_RB)))*B

            if (gamaAngle[y, x] <= localQuota):
                gradDirCosts[y, x] = 255
            else:
                gradDirCosts[y, x] = 0

    # gxy = cv2.Sobel(cameraImage, cv2.CV_32F, 1, 1)
    # gradientDirectionsGxy = np.arctan(gxy)
    # cv2.imshow('x2', gradientDirectionsGxy)

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Exiting computeGradientDirectionCost") 

    return gradDirCosts

def computeDisparityFeatureCost(disparityImage):
    """
    For convenience, the disparity values of central pixel and its eight neighbors
    are represented as follow:
    Let D(i,j) denotate Disparity Value of pixel P(i,j).
        b0 = D(i-1, j-1)
        b1 = D(i-1,j)
        b2 = D(i-1,j+1)
        b3 = D(i, j-1)
        b4 = D(i,j)
        b5 = D(i,j+1)
        b6 = D(i+1, j-1)
        b7 = D(i+1, j+1)

        Then F(P(i,j)) = sum(k=0->7){Ck * 2^k}
    """
    def distributeDisparityValueNeighboors(x, y):
        """
        Helper function to distribute the disparity cost of 8 neighbors.
        """
        assert x > 0 and y > 0

        b0 = int(disparityImage[y-1, x-1])
        b1 = int(disparityImage[y, x-1])
        b2 = int(disparityImage[y+1, x-1])
        b3 = int(disparityImage[y-1, x])
        b4 = int(disparityImage[y, x])
        b5 = int(disparityImage[y+1, x])
        b6 = int(disparityImage[y-1, x+1])
        b7 = int(disparityImage[y, x+1])
        b8 = int(disparityImage[y+1, x+1])

        return (b0, b1, b2, b3, b4, b5, b6, b7, b8)

    def calculateConstantsC(b0, b1, b2, b3, b4, b5, b6, b7, b8):
        """
        Helper method to compute Disparity Feature Constants.
            c0, c1, c2, c3, c4, c5, c6, c7.
        """
        c0 = (0, 1)[(b0 + b1 + b2) < (b3 + b4 + b5)]
        c1 = (0, 1)[(b3 + b4 + b5) < (b6 + b7 + b8)]
        c2 = (0, 1)[b1 < b4]
        c3 = (0, 1)[b4 < b7]
        c4 = (0, 1)[b0 < b4]
        c5 = (0, 1)[b2 < b4]
        c6 = (0, 1)[b4 < b6]
        c7 = (0, 1)[b4 < b8]

        return ([c0, c1, c2, c3, c4, c5, c6, c7])

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Entered computeDisparityFeatureCost") 

    disparityImageCost = np.asarray(disparityImage.copy())
    height, width = disparityImage.shape
    # print (height, width)
    # For all pixels in img
    for x in range(1, width-1):
        for y in range(1, height-1):
            b0, b1, b2, b3, b4, b5, b6, b7, b8 = distributeDisparityValueNeighboors(
                x, y)
            c_consts = calculateConstantsC(b0, b1, b2, b3, b4, b5, b6, b7, b8)

            # sum the cost of each neighboor
            pixelFeatureCost = 0
            for k in range(8):
                pixelFeatureCost = pixelFeatureCost + c_consts[k]*2**k

            # update the value
            disparityImageCost[y, x] = pixelFeatureCost

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Exiting computeDisparityFeatureCost") 

    return disparityImageCost

def computeFlatnessCost(disparityFrame, k, b):
    """
    Compues Flatness Cost.
    """

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Entered computeFlatnessCost") 

    flatnesFrameCost = np.asarray(disparityFrame.copy())
    height, width = disparityFrame.shape
    
    ALPHA = 0.13
    for y in range(height):
        for x in range(width):
            xLamda = 1/k * (y - b)
            localQuota = abs(disparityFrame[y,x] - xLamda)
            
            if (localQuota <= xLamda*ALPHA):
                flatnesFrameCost[y,x] = 255
            else:
                flatnesFrameCost[y,x] = 0

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Exiting computeFlatnessCost") 
    
    return flatnesFrameCost

def computeOrientationLinkCost(pu_x, pu_y, pv_x, pv_y, gx,gy):
    """
    Link cost = 2/(3pi) * (arccos(Opu(pu,pv)) + arccos(Opv(pu,pv)))
    Opu(pu,pv) = Opu dot L(pu,pv)
    Opv(pu,pv) = Opv dot L(pu,pv)
    """
    def getCost(U, V):
        # print(U,V)
        return (3/(2 * np.pi)) * (np.arccos(U) + np.arccos(V))

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Entered computeOrientationLinkCost")

    point_pu = np.array([pu_y, pu_x])
    point_pv = np.array([pv_x, pv_y])
    Opu = np.array([gy[pu_y, pu_x], -gx[pu_y, pu_x]])
    Opv = np.array([gy[pv_y, pv_x], -gx[pv_y, pv_x]])
    # print(point_pv - point_pu)
    if (np.dot(Opu, point_pv - point_pu) >= 0):
        linkVector = point_pv - point_pu
    else:
        linkVector = point_pu - point_pv 

    linkCost = getCost(np.dot(Opu, linkVector), np.dot(Opv, linkVector))

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Exiting computeOrientationLinkCost")

    return linkCost

def computeGradientCost(gx,gy):
    """
    Gradient Cost for Image pixel Pv is:
        ---> f(Pv) = 1 - (G(Pv) / Gmax)
        where: 
            Gmax = maximum gradient magnitude of all pixels
            G(Pv) = Grdient Magnitude at Pixel Pv.

    """
    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Entered computeGradientCost")
    
    # get magnitude for each pixel and Maximum gradient
    magnitude = np.asarray(cv2.magnitude(gx, gy))
    maximumGradient = magnitude.max()

    # get the gradientCosts Matrix based on its function
    gradientsCostMatrix = np.asarray(1 - magnitude/maximumGradient)

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Exiting computeGradientCost")

    return gradientsCostMatrix

def computeSobelDerivatives(frame):
    """
    Returns Gx and Gy Sobel Derivatives Normalized.
    """
    # Get x and y gradients from Camera Image
    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)

    # normalize
    max_x = gx.max()
    max_y = gy.max()

    gx = gx/max_x
    gy = gy/max_y

    return gx, gy