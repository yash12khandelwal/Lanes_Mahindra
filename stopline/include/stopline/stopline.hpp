#ifndef _stopline_HPP
#define _stopline_HPP

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/Image.h"
#include <message_filters/subscriber.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <eigen3/Eigen/Dense>
#include "geometry_msgs/Polygon.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;


int wi=5;      				//width of stop line
float lateral_thresh = 10;  //allowance for line to be lateral(around 90)
int lat_cluster_thresh = 10;//seperation between x distances of 2 clusters for them to be considered as same cluster
float g_thresh = 0.997; 	//sin of angle between stopline and horizon line to be considered as stopline
int area_thresh = 20; 		//Below this area dont consider clusters

geometry_msgs::Pose2D singlePos;



//Structure to store each blobs data
typedef struct Cluster_Data{
    float g=0;
    bool l_thresh=0;
    int centroid_x=0;
    int centroid_y=0;
    double theta=0;
    float length=0;
}CD;

int isvalid(Mat A, int i, int j);
float* PCa(vector<vector<Point> > data);
void gradient_thresholding(Mat img, Mat dst);
void gradient_thresholding_horizontal(Mat image,Mat original, Mat dst);
int colour(Mat tp,Mat dst_labels, int nlabels);
void merge(int arr[][3], int l, int m, int r) ;
void mergeSort(int arr[][3], int l, int r) ;
int x1,y_1,x2,y2,x3,y3,x4,y4;



#endif