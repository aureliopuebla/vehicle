#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_broadcaster.h>
#include <time.h>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>

#define DATA_DIR "/home/aurelio/KITTI Dataset/2011_09_26_drive_0091_sync/"
#define LEFT_DIR "image_02/data/"
#define DISP_DIR "proj_depth/groundtruth/image_02/"
#define OXTS_DIR "oxts/data/"
#define FIRST_IDX 5
#define LAST_IDX 334
#define DEPTH_TO_DISP 128*65536
#define FRAME_ID "multisense/left_camera_optical_frame"


void getGPS(int idx, ros::Time timestamp, sensor_msgs::NavSatFix *gpsFixMsg) {
  std::stringstream ss;
  ss << DATA_DIR << OXTS_DIR;
  ss << std::setfill('0') << std::setw(10) << idx << ".txt";

  std::ifstream oxtsFile(ss.str());
  std::string line = "";
  getline(oxtsFile, line);

  boost::char_separator<char> sep{" "};
  boost::tokenizer<boost::char_separator<char> > tok(line, sep);
  std::vector<std::string> s(tok.begin(), tok.end());

  gpsFixMsg->header.frame_id = FRAME_ID;
  gpsFixMsg->header.stamp = timestamp;

  gpsFixMsg->latitude  = boost::lexical_cast<double>(s[0]);
  gpsFixMsg->longitude = boost::lexical_cast<double>(s[1]);
  gpsFixMsg->altitude  = boost::lexical_cast<double>(s[2]);

  gpsFixMsg->position_covariance_type =
      sensor_msgs::NavSatFix::COVARIANCE_TYPE_APPROXIMATED;
  for (int i = 0; i < 9; i++)
    gpsFixMsg->position_covariance[i] = 0.0f;

  gpsFixMsg->position_covariance[0] = boost::lexical_cast<double>(s[23]);
  gpsFixMsg->position_covariance[4] = boost::lexical_cast<double>(s[23]);
  gpsFixMsg->position_covariance[8] = boost::lexical_cast<double>(s[23]);

  gpsFixMsg->status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;
  gpsFixMsg->status.status  = sensor_msgs::NavSatStatus::STATUS_GBAS_FIX;
}


void getIMU(int idx, ros::Time timestamp, sensor_msgs::Imu *imuMsg) {
  std::stringstream ss;
  ss << DATA_DIR << OXTS_DIR;
  ss << std::setfill('0') << std::setw(10) << idx << ".txt";

  std::ifstream oxtsFile(ss.str());
  std::string line = "";
  getline(oxtsFile, line);

  boost::char_separator<char> sep{" "};
  boost::tokenizer<boost::char_separator<char> > tok(line, sep);
  std::vector<std::string> s(tok.begin(), tok.end());

  imuMsg->header.frame_id = FRAME_ID;
  imuMsg->header.stamp = timestamp;

  //    - ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
  //    - ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
  //    - az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
  imuMsg->linear_acceleration.x = boost::lexical_cast<double>(s[11]);
  imuMsg->linear_acceleration.y = boost::lexical_cast<double>(s[12]);
  imuMsg->linear_acceleration.z = boost::lexical_cast<double>(s[13]);

  //    - vf:      forward velocity, i.e. parallel to earth-surface (m/s)
  //    - vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
  //    - vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
  imuMsg->angular_velocity.x = boost::lexical_cast<double>(s[8]);
  imuMsg->angular_velocity.y = boost::lexical_cast<double>(s[9]);
  imuMsg->angular_velocity.z = boost::lexical_cast<double>(s[10]);

  //    - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
  //    - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
  //    - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
  tf::Quaternion q = tf::createQuaternionFromRPY(
      boost::lexical_cast<double>(s[3]),
      boost::lexical_cast<double>(s[4]),
      boost::lexical_cast<double>(s[5]));
  imuMsg->orientation.x = q.getX();
  imuMsg->orientation.y = q.getY();
  imuMsg->orientation.z = q.getZ();
  imuMsg->orientation.w = q.getW();

  // Broadcast Orientation Transform
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
  transform.setRotation(q);
  br.sendTransform(
      tf::StampedTransform(transform, timestamp, "map", "base_link"));
}


void getImages(int idx,
               sensor_msgs::CameraInfo *camInfoMsg,
               sensor_msgs::Image *leftImageMsg,
               sensor_msgs::Image *depthImageMsg,
               sensor_msgs::Image *dispImageMsg) {
  std::stringstream ssl, ssd;
  ssl << DATA_DIR << LEFT_DIR;
  ssl << std::setfill('0') << std::setw(10) << idx << ".png";
  ssd << DATA_DIR << DISP_DIR;
  ssd << std::setfill('0') << std::setw(10) << idx << ".png";

  cv::Mat leftImage = cv::imread(ssl.str(), cv::IMREAD_UNCHANGED);
  // KIITI depth images are stored as uint16 with a 1/256 factor to meters.
  cv::Mat depthImage, temp = cv::imread(ssd.str(), cv::IMREAD_UNCHANGED);
  temp.convertTo(depthImage, CV_32FC1);
  depthImage /= 256.0;
  // Convert uint16 depth image (temp) to disp by a factor of the inverse.
  cv::Mat dispImage = cv::imread(ssd.str(), cv::IMREAD_UNCHANGED);
  cv::threshold(DEPTH_TO_DISP / (temp + 1), // Convert to Disparity.
                dispImage, // dst
                65535-1, // threshold, eliminates invalid depths/disps.
                0,
                cv::THRESH_TOZERO_INV);
  // Extend sparse values to simulate the Multisense density.
  //cv::Mat count = (dispImage != 0) / 255;
  //count.convertTo(count, CV_16UC1);
  //cv::boxFilter(count, count, -1 /*ddepth*/, cv::Size(3, 3),
  //              cv::Point(-1,-1), false/*normalize*/);
  //cv::blur(dispImage, dispImage, cv::Size(3, 3));
  //dispImage = dispImage.mul(count);

  cv_bridge::CvImage(camInfoMsg->header, "bgr8", leftImage)
      .toImageMsg(*leftImageMsg);
  cv_bridge::CvImage(camInfoMsg->header, "32FC1", depthImage)
      .toImageMsg(*depthImageMsg);
  cv_bridge::CvImage(camInfoMsg->header, "16UC1", dispImage)
      .toImageMsg(*dispImageMsg);
}


int main(int argc, char** argv) {
  ros::init(argc, argv, "sensors_sim");
  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);
  image_transport::CameraPublisher leftPub =
      it.advertiseCamera("camera/left/image_rect", 1);
  image_transport::CameraPublisher depthPub =
      it.advertiseCamera("camera/depth/image_rect", 1);
  image_transport::CameraPublisher dispPub =
      it.advertiseCamera("camera/disp/image_rect", 1);
  ros::Publisher gpsPub =
      nh.advertise<sensor_msgs::NavSatFix>("oxts/gps", 1);
  ros::Publisher imuPub =
      nh.advertise<sensor_msgs::Imu>("oxts/imu", 1);

  // Camera Parameters extracted from calib_cam_to_cam.txt
  sensor_msgs::CameraInfo camInfoMsg;
  camInfoMsg.header.frame_id = FRAME_ID;
  camInfoMsg.height = 375;
  camInfoMsg.width = 1242;
  camInfoMsg.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
  camInfoMsg.D = {
      -3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02};
  camInfoMsg.K = {9.597910e+02, 0.000000e+00, 6.960217e+02,
                  0.000000e+00, 9.569251e+02, 2.241806e+02,
                  0.000000e+00, 0.000000e+00, 1.000000e+00};
  camInfoMsg.R = {9.998817e-01, 1.511453e-02, -2.841595e-03,
                  -1.511724e-02, 9.998853e-01, -9.338510e-04,
                  2.827154e-03, 9.766976e-04, 9.999955e-01};
  camInfoMsg.P = {7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
                  0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
                  0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03};

  sensor_msgs::Image leftImageMsg, depthImageMsg, dispImageMsg;
  sensor_msgs::NavSatFix gpsFixMsg;
  sensor_msgs::Imu imuMsg;

  ros::Rate loop_rate(10);
  while (nh.ok()) {
    for (int idx = FIRST_IDX; nh.ok() && idx <= LAST_IDX; idx++) {
      ros::Time timestamp = ros::Time::now();

      getImages(idx, &camInfoMsg, &leftImageMsg, &depthImageMsg, &dispImageMsg);
      leftPub.publish(leftImageMsg, camInfoMsg, timestamp);
      depthPub.publish(depthImageMsg, camInfoMsg, timestamp);
      dispPub.publish(dispImageMsg, camInfoMsg, timestamp);

      getGPS(idx, timestamp, &gpsFixMsg);
      gpsPub.publish(gpsFixMsg);

      getIMU(idx, timestamp, &imuMsg);
      imuPub.publish(imuMsg);

      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0; 
}
