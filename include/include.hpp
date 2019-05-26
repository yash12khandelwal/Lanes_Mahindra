#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <iostream>
#include <ros/ros.h>
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Vector3.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include "geometry_msgs/Polygon.h"
#include "../src/lsd.cpp"
#include "../src/laneDetector_utils.cpp"
#include "houghP.hpp"
#include "laneDetector_utils.hpp"
#include "lsd.h"
#include <string>
// #include "armadillo"
#include "ransac.hpp"
// #include "mlesac.hpp"
#include <nav_msgs/Odometry.h>
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include <sensor_msgs/LaserScan.h>
#include <dynamic_reconfigure/server.h>
#include <lanes/TutorialsConfig.h>

model lanes, previous;

vector<Point> cost_map_lanes;
sensor_msgs::LaserScan scan_global;

void callback(node::TutorialsConfig &config, uint32_t level)
Mat findIntensityMaxima(Mat img)
Mat findEdgeFeatures(Mat img, bool top_edges)
Mat fit_ransac(Mat all_features)
void publish_lanes(Mat lanes_by_ransac)
vector<Point> detect_lanes(Mat img)
void imageCb(const sensor_msgs::ImageConstPtr& msg);
sensor_msgs::LaserScan imageConvert(Mat image);

int flag=1;

using namespace std;
using namespace cv;
using namespace ros;