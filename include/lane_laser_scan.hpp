#ifndef LANE_LASER_SCAN
#define LANE_LASER_SCAN

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cmath>
#include <limits>
#include <iostream>

using namespace ros;
using namespace std;
using namespace cv;


sensor_msgs::LaserScan imageConvert(Mat img)
{
    cvtColor(img,img,CV_BGR2GRAY);

    int row = img.rows;
    int col = img.cols;
    sensor_msgs::LaserScan scan;
    scan.angle_min = -CV_PI/2;
    scan.angle_max = CV_PI/2;
    scan.angle_increment = CV_PI/bins;
    double inf = std::numeric_limits<double>::infinity();
    scan.range_max = inf; 
    
    scan.header.frame_id = "laser";

    for (int i=0;i<bins;i++)
    {
        scan.ranges.push_back(scan.range_max);
    }

    scan.range_max = 80;
    for(int i = 0; i < row; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            if(img.at<uchar>(i, j) > 0)
            {
                float a = (col/2 - j)/pixelPerMeter;
                float b = (row - i)/pixelPerMeter + yshift;

                double angle = atan(a/b);

                double r = sqrt(a*a  + b*b);

                int k = (angle - scan.angle_min)/(scan.angle_increment);
                scan.ranges[bins-k-1] = r ;
            }
        }
    }

    return scan;
}


#endif
