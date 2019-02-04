#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sstream>
#include <iomanip>

#define DATA_DIR "/home/aurelio/KITTI Dataset/2011_09_26_drive_0091_sync/"
#define LEFT_DIR "image_02/data/"
#define DISP_DIR "proj_depth/groundtruth/image_02/"
#define FIRST_IMAGE_IDX 5
#define LAST_IMAGE_IDX 334
#define DEPTH_TO_DISP 128*65536
#define FRAME_ID "multisense/left_camera_optical_frame"

int main(int argc, char** argv) {
  ros::init(argc, argv, "stereo");
  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);
  image_transport::CameraPublisher leftPub =
      it.advertiseCamera("camera/left/image", 1);
  image_transport::CameraPublisher depthPub =
      it.advertiseCamera("camera/depth/image", 1);
  image_transport::CameraPublisher dispPub =
      it.advertiseCamera("camera/disp/image", 1);

  sensor_msgs::Image leftImageMsg, depthImageMsg, dispImageMsg;
  // Camera Parameters extracted from calib_cam_to_cam.txt
  sensor_msgs::CameraInfo camInfoMsg;
  camInfoMsg.header.frame_id = FRAME_ID;
  camInfoMsg.height = 1242;
  camInfoMsg.width = 375;
  /*camInfoMsg.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
  camInfoMsg.D = {
      -3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02};*/
  camInfoMsg.K = {9.597910e+02, 0.000000e+00, 6.960217e+02,
                  0.000000e+00, 9.569251e+02, 2.241806e+02,
                  0.000000e+00, 0.000000e+00, 1.000000e+00};
  camInfoMsg.R = {9.998817e-01, 1.511453e-02, -2.841595e-03,
                  -1.511724e-02, 9.998853e-01, -9.338510e-04,
                  2.827154e-03, 9.766976e-04, 9.999955e-01};
  camInfoMsg.P = {7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
                  0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
                  0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03};

  ros::Rate loop_rate(5);
  while (nh.ok()) {
    for (int i = FIRST_IMAGE_IDX; nh.ok() && i <= LAST_IMAGE_IDX; i++) {
      ros::Time currentTimestamp = ros::Time::now();

      std::stringstream ssl, ssd;
      ssl << DATA_DIR << LEFT_DIR; 
      ssl << std::setfill('0') << std::setw(10) << i << ".png";
      ssd << DATA_DIR << DISP_DIR; 
      ssd << std::setfill('0') << std::setw(10) << i << ".png";

      cv::Mat leftImage = cv::imread(ssl.str(), cv::IMREAD_UNCHANGED);
      cv::Mat depthImage = cv::imread(ssd.str(), cv::IMREAD_UNCHANGED);
      /*cv::threshold(DEPTH_TO_DISP / (dispImage + 1), // Convert to Disparity.
                    dispImage, // dst
                    65535-1, // thresshold, eliminates invalid depths/disps.
                    0, 
                    cv::THRESH_TOZERO_INV);  !!!!!!!!!!!!!!!!!!!! */
      // Extend sparse values to simulate the Multisense density.
      //cv::Mat count = (dispImage != 0) / 255;
      //count.convertTo(count, CV_16UC1);
      //cv::boxFilter(count, count, -1 /*ddepth*/, cv::Size(3, 3),
      //              cv::Point(-1,-1), false/*normalize*/);
      //cv::blur(dispImage, dispImage, cv::Size(3, 3));
      //dispImage = dispImage.mul(count);

      cv_bridge::CvImage(camInfoMsg.header, "bgr8", leftImage)
          .toImageMsg(leftImageMsg);
      cv_bridge::CvImage(camInfoMsg.header, "16UC1", depthImage)
          .toImageMsg(depthImageMsg);

      leftPub.publish(leftImageMsg, camInfoMsg, currentTimestamp);
      depthPub.publish(depthImageMsg, camInfoMsg, currentTimestamp);

      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0; 
}
