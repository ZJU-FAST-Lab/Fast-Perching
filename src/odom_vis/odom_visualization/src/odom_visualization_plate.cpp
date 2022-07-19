#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Geometry>

static double height_, width_, platform_r_;

ros::Publisher polygen_pub_, cylinder_pub_;

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
  Eigen::Vector3d odom_p(msg->pose.pose.position.x,
                         msg->pose.pose.position.y,
                         msg->pose.pose.position.z);
  Eigen::Quaterniond odom_q(msg->pose.pose.orientation.w,
                            msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y,
                            msg->pose.pose.orientation.z);

  Eigen::MatrixXd R = odom_q.toRotationMatrix();
  double dx = height_ / 2;
  double dy = width_ / 2;

  visualization_msgs::Marker polygon;
  polygon.header = msg->header;
  polygon.type = visualization_msgs::Marker::CUBE;
  polygon.pose = msg->pose.pose;
  polygon.scale.x = height_ / 2;
  polygon.scale.y = width_ / 2;
  polygon.scale.z = 0.01;
  polygon.color.a = 0.5;
  polygon.color.r = 0.0;
  polygon.color.g = 0.0;
  polygon.color.b = 1.0;
  polygen_pub_.publish(polygon);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "odom_visualization_plate");
  ros::NodeHandle nh("~");
  nh.getParam("height", height_);
  nh.getParam("width", width_);
  nh.getParam("platform_r", platform_r_);

  ros::Subscriber sub_odom = nh.subscribe("odom", 100, odom_callback);
  polygen_pub_ = nh.advertise<visualization_msgs::Marker>("polygon", 100, true);

  ros::spin();
  return 0;
}
