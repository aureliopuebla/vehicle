#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sstream>
#include <iomanip>

#define DATA_DIR "/home/aurelio/KITTI Dataset/2011_09_26_drive_0091_sync/"
#define LEFT_DIR "image_02/data/"
#define DISP_DIR "proj_depth/groundtruth/image_02/"
#define FIRST_IMAGE_IDX 5
#define LAST_IMAGE_IDX 334

int main(int argc, char** argv) {
  ros::init(argc, argv, "stereo");
  ros::NodeHandle nh;
  ros::Publisher leftPub = nh.advertise<sensor_msgs::Image>("camera/image", 1);
  ros::Publisher dispPub = nh.advertise<sensor_msgs::Image>("camera/disp", 1);

  ros::Rate loop_rate(5);
  while (nh.ok()) {
    for (int i = FIRST_IMAGE_IDX; nh.ok() && i <= LAST_IMAGE_IDX; i++) {
      std::stringstream ssl, ssd;
      ssl << DATA_DIR << LEFT_DIR; 
      ssl << std::setfill('0') << std::setw(10) << i << ".png";
      ssd << DATA_DIR << DISP_DIR; 
      ssd << std::setfill('0') << std::setw(10) << i << ".png";

      cv::Mat leftImage = cv::imread(ssl.str(), cv::IMREAD_UNCHANGED);
      cv::Mat depthImage = cv::imread(ssd.str(), cv::IMREAD_UNCHANGED);

      sensor_msgs::ImagePtr leftMsg = cv_bridge::CvImage(
          std_msgs::Header(), "bgr8", leftImage).toImageMsg();
      sensor_msgs::ImagePtr dispMsg = cv_bridge::CvImage(
          std_msgs::Header(), "16UC1", depthImage).toImageMsg();
    
      leftPub.publish(leftMsg);
      dispPub.publish(dispMsg);
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0; 
}
