#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sstream>
#include <iomanip>

#define DATA_DIR "/home/aurelio/vehicle_ws/data/"

int main(int argc, char** argv) {
  ros::init(argc, argv, "stereo");
  ros::NodeHandle nh;
  ros::Publisher leftPub = nh.advertise<sensor_msgs::Image>("camera/image", 1);
  ros::Publisher dispPub = nh.advertise<sensor_msgs::Image>("camera/disp", 1);

  ros::Rate loop_rate(5);
  while (nh.ok()) {
    for (int i = 0; nh.ok() && i < 200; i++) {
      std::stringstream ssl, ssd;
      ssl << DATA_DIR << "left/" << std::setfill('0') << std::setw(6) << i << "_10.png";
      ssd << DATA_DIR << "disp/" << std::setfill('0') << std::setw(6) << i << "_10.png";

      cv::Mat leftImage = cv::imread(ssl.str(), CV_LOAD_IMAGE_COLOR);
      cv::Mat dispImage = cv::imread(ssd.str(), CV_LOAD_IMAGE_GRAYSCALE);
      cv::waitKey(30);
      sensor_msgs::ImagePtr leftMsg = cv_bridge::CvImage(
          std_msgs::Header(), "bgr8", leftImage).toImageMsg();
      sensor_msgs::ImagePtr dispMsg = cv_bridge::CvImage(
          std_msgs::Header(), "mono8", dispImage).toImageMsg();
    
      leftPub.publish(leftMsg);
      dispPub.publish(dispMsg);
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0; 
}
