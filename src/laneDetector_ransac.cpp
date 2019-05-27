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
// #include "armadillo"
#include <nav_msgs/Odometry.h>
#include "tf/tf.h"
#include <sensor_msgs/LaserScan.h>
#include <dynamic_reconfigure/server.h>
#include <lanes/TutorialsConfig.h>


#include <params.hpp>
#include "lsd.cpp"
#include "laneDetector_utils.cpp"
#include "houghP.hpp"
#include "laneDetector_utils.hpp"
#include "lsd.h"
#include <ransac.hpp>
// #include "mlesac.hpp"
#include <matrixTransformation.hpp>

Parabola lanes, previous;

vector<Point> cost_map_lanes;
sensor_msgs::LaserScan scan_global;

void callback(node::TutorialsConfig &config, uint32_t level);
Mat findIntensityMaxima(Mat img);
Mat findEdgeFeatures(Mat img, bool top_edges);
Mat find_all_features(Mat boundary, Mat intensityMaxima, Mat edgeFeature);
Mat fit_ransac(Mat all_features);
void publish_lanes(Mat lanes_by_ransac);
void detect_lanes(Mat img);
void imageCb(const sensor_msgs::ImageConstPtr& msg);
sensor_msgs::LaserScan imageConvert(Mat image);

int flag=1;

using namespace std;
using namespace cv;
using namespace ros;

Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
int size_X = 800;
int size_Y = 1000;

void callback(node::TutorialsConfig &config, uint32_t level)
{
    is_debug = config.is_debug;

    // ransac parameters
	iteration = config.iteration;
	maxDist = config.maxDist;
	minLaneInlier = config.minLaneInlier;
	minPointsForRANSAC = config.minPointsForRANSAC;
	grid_size = config.grid_size;

	// publish parameters
	pixelPerMeter = config.pixelPerMeter;

    // edgeFeatures parameters
	horizon = config.horizon;
	horizon_offset = config.horizon_offset;

	transformedPoints0_lowerbound = config.transformedPoints0_lowerbound;
	transformedPoints0_upperbound = config.transformedPoints0_upperbound;
	point1_y = config.point1_y;
	point2_x = config.point2_x;

	// intensity maxima parameters
	h = config.h;
	w = config.w;
	variance = config.variance;
	hysterisThreshold_min = config.hysterisThreshold_min;
	hysterisThreshold_max = config.hysterisThreshold_max;

	// distance between first point of image and lidar
	yshift = config.yshift;

	// region of interest in all_features
	y = config.y;
	lane_width = config.lane_width;
	k1 = config.k1;
	k2 = config.k2;

	// blue channel image parameters
	medianBlurkernel = config.medianBlurkernel;
    neighbourhoodSize = config.neighbourhoodSize;
    constantSubtracted = config.constantSubtracted;
}

Mat findIntensityMaxima(Mat img)
{
    Mat topview = top_view(img, ::transform, size_X, size_Y);

    GaussianBlur(topview, topview, Size(5, 15), 0, 0);
    blur(topview, topview, Size(25,25));

    if(is_debug == true){
    	namedWindow("topview", WINDOW_NORMAL);
    	imshow("topview", topview);
    }

    cvtColor(topview, topview, CV_BGR2GRAY);
    medianBlur(topview, topview, 3);

    // template matching
    Mat t0, t1, t2, t3, t4, t5, t6;
    matchTemplate(topview, getTemplateX2(2.1, h, w, -10), t0, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2.1, h, w,   0), t1, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2.1, h, w,  10), t2, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2.1, h, w,  -20), t3, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2.1, h, w,  +20), t4, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2.1, h, w,  +30), t5, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2.1, h, w,  -30), t6, CV_TM_CCOEFF_NORMED);

    Mat t = t0-t0;
    for(int i=0;i<t.rows;i++)
        for(int j=0;j<t.cols;j++)
        {
            t.at<float>(i, j) = max( t4.at<float>(i, j), max(t3.at<float>(i, j), max(t0.at<float>(i, j), max(t1.at<float>(i, j), t2.at<float>(i, j)))));
            t.at<float>(i, j) = max( t.at<float>(i, j), max(t5.at<float>(i, j), t6.at<float>(i, j)) );
        }

    // ########threshold###########
    hysterisThreshold(t, topview, hysterisThreshold_min, hysterisThreshold_max);

    Mat result=Mat(topview.rows+h-1, topview.cols+w-1,CV_8UC1, Scalar(0));
    for(int i=0;i<topview.rows;i++)
        for(int j=0;j<topview.cols;j++)
            result.at<uchar>(i+(h-1)/2,j+(w-1)/2)=255*topview.at<float>(i,j);


    if(is_debug==true)
    {
        namedWindow("intensityMaxima", WINDOW_NORMAL);
        imshow("intensityMaxima", result);
    }

    return result;
}

Mat findEdgeFeatures(Mat img, bool top_edges)
{
    vector<Vec4i> lines, lines_top;
    vector<int> line_lens;
    Mat edges;

    // this will create lines using lsd 
    if(top_edges == false)
    {
        
        Mat src = img;
        Mat tmp, src_gray;
        cvtColor(src, tmp, CV_RGB2GRAY);
        tmp.convertTo(src_gray, CV_64FC1);

        int cols  = src_gray.cols;
        int rows = src_gray.rows;
        image_double image = new_image_double(cols, rows);
        image->data = src_gray.ptr<double>(0);
        ntuple_list ntl = lsd(image);
        Mat lsd = Mat::zeros(rows, cols, CV_8UC1);
        Point pt1, pt2;
        for (int j = 0; j != ntl->size ; ++j)
        {
            Vec4i t;

            pt1.x = int(ntl->values[0 + j * ntl->dim]);
            pt1.y = int(ntl->values[1 + j * ntl->dim]);
            pt2.x = int(ntl->values[2 + j * ntl->dim]);
            pt2.y = int(ntl->values[3 + j * ntl->dim]);
            t[0]=pt1.x;
            t[1]=pt1.y;
            t[2]=pt2.x;
            t[3]=pt2.y;
            lines.push_back(t);
            int width = int(ntl->values[4 + j * ntl->dim]);

            line(lsd, pt1, pt2, Scalar(255), width + 1, CV_AA);
        }
        free_ntuple_list(ntl);
        edges=lsd;
    }

    // this will create lines using canny
    else
    {
        Mat topview = top_view(img, ::transform, size_X, size_Y);
        Canny(topview, edges, 200, 300);
        HoughLinesP(edges, lines_top, 1, CV_PI/180, 60, 60, 50);
        transformLines(lines_top, lines, ::transform.inv());
    }

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        Point2f inputPoints[2], transformedPoints[2];

        inputPoints[0] = Point2f( l[0], l[1] );
        inputPoints[1] = Point2f( l[2], l[3] ); 
        transformPoints(inputPoints, transformedPoints, ::transform, 2);

        if(transformedPoints[0].x<transformedPoints0_lowerbound || transformedPoints[0].x>transformedPoints0_upperbound || transformedPoints[1].x<transformedPoints0_lowerbound || transformedPoints[1].x>transformedPoints0_upperbound || l[1] < point1_y || l[2] < point2_x)
        {
            lines.erase(lines.begin() + i);
            i--;
        }
    }

    int is_lane[lines.size()];
    for(int i=0;i<lines.size();i++)
        is_lane[i]=0;

    for(size_t i=0;i<lines.size();i++)
        for(size_t j=0;j<lines.size();j++)
            if(abs(findIntersection(lines[i],lines[j])-horizon)<horizon_offset)
            {
                is_lane[i]+=1;
                is_lane[j]+=1;
            }

    vector<Vec4i> lane_lines, lane_lines_top;

    for(int i=0;i<lines.size();i++)
        if(is_lane[i]>10)
            lane_lines.push_back(lines[i]);

    transformLines(lane_lines, lane_lines_top, ::transform);
    Mat edgeFeatures;
    Mat filtered_lines = Mat(img.size(),CV_8UC1, Scalar(0));
    for( size_t i = 0; i < lane_lines.size(); i++ )
    {
        Vec4i l = lane_lines[i];
        line(filtered_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 5, CV_AA);
    }
    edgeFeatures = top_view(filtered_lines, ::transform, size_X, size_Y);

    if(is_debug==true)
    {
        namedWindow("edges", WINDOW_NORMAL);
        namedWindow("edgeFeatures", WINDOW_NORMAL);
        namedWindow("filtered_lines", WINDOW_NORMAL);

        imshow("edges", edges);
        imshow("edgeFeatures", edgeFeatures);
        imshow("filtered_lines", filtered_lines);
    }

    return edgeFeatures;
}

Mat find_all_features(Mat boundary, Mat intensityMaxima, Mat edgeFeature)
{
	Mat all_features = Mat(size_Y, size_X,CV_8UC1, Scalar(0));

    medianBlur(intensityMaxima,intensityMaxima, 5);

    for(int i=0; i<edgeFeature.rows; i++)
    {
        for(int l = 0; l < 2; l++ )
        {
            for(int offset = lane_width/2 - k1; offset < lane_width/2 + k2; offset++)
            {
                if( l > 0 )
                    offset = -offset;
                
                int j = (int)(y+offset);

                all_features.at<uchar>(i,j)=30;
                if(intensityMaxima.at<uchar>(i, j)>8)
                {
                    all_features.at<uchar>(i,j) = {255};
                }
                
                if(edgeFeature.at<uchar>(i, j)>5)
                {
                    all_features.at<uchar>(i, j) = {255};
                }
                
                if(boundary.at<uchar>(i, j)>5)
                {
                    all_features.at<uchar>(i, j) = {255};
                }
                
                if( l > 0 )
                    offset = -offset;

            }
        }
    }

    namedWindow("all_features", WINDOW_NORMAL);
    imshow("all_features", all_features);

    return all_features;
}

Mat fit_ransac(Mat all_features)
{
	Mat all_features_frontview = front_view(all_features, ::transform);

    lanes = getRansacModel(all_features_frontview, previous);
    previous=lanes;

    Mat fitLanes = drawLanes(all_features_frontview, lanes);
    // Mat originalLanes = drawLanes(all_features_frontview, lanes);

    namedWindow("lanes_by_ransac", WINDOW_NORMAL);
    imshow("lanes_by_ransac", fitLanes);

    return fitLanes;
}

void publish_lanes(Mat lanes_by_ransac)
{
	Mat lanes_by_ransac_topview = top_view(lanes_by_ransac, ::transform, size_X, size_Y);

    scan_global = imageConvert(lanes_by_ransac_topview); 
}

Mat blueChannelProcessing(Mat img)
{
    Mat channels[3];
    split(img, channels);
    Mat b = channels[0];

    GaussianBlur(b , b, Size( 9, 9), 0, 0);
    adaptiveThreshold(b,b,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,neighbourhoodSize, constantSubtracted);
    medianBlur(b,b,medianBlurkernel);

    if(is_debug){
    	namedWindow("blue channel image", WINDOW_NORMAL);
    	imshow("blue channel image", b);
    }

    return b;
}

void detect_lanes(Mat img)
{
    if(is_debug==true)
    {
        namedWindow("original", WINDOW_NORMAL);
        imshow("original", img);
    }

    // initialize boundary with a matrix of (800*400)
    Mat boundary = Mat(size_Y, size_X, CV_8UC1, Scalar(0));

    // intenity maximum image made
    Mat intensityMaxima = findIntensityMaxima(img);

    // image with edge features made
    Mat edgeFeature = findEdgeFeatures(img, false);

    // blue channel image
    Mat b = blueChannelProcessing(img);

    // curve fit on the basis of orignal image, maxima intensity image and edgeFeature image
    Mat all_features = find_all_features(boundary, intensityMaxima, edgeFeature);

    Mat lanes_by_ransac = fit_ransac(all_features);

    publish_lanes(lanes_by_ransac);
}

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
    flag=1;
    Mat img;
    cv_bridge::CvImagePtr cv_ptr;

    cout<<"in callback"<<endl;

    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        img = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if( !img.data ) { printf("Error loading A \n"); return ; }

    detect_lanes(img);
}

sensor_msgs::LaserScan imageConvert(Mat img)
{
    int bins = 1080;
    int row = img.rows;
	int col = img.cols;
	sensor_msgs::LaserScan scan;
	scan.angle_min = -CV_PI/2;
	scan.angle_max = CV_PI/2;
	scan.angle_increment = CV_PI/bins;
	double inf = std::numeric_limits<double>::infinity();
	scan.range_max = inf; 
	
	scan.header.frame_id = "laser";

	cvtColor(img,img,CV_BGR2GRAY);    

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

int main(int argc, char **argv)
{
    init(argc, argv, "lanes");
    NodeHandle nh_;
    image_transport::ImageTransport it_(nh_);

    dynamic_reconfigure::Server<node::TutorialsConfig> server;
    dynamic_reconfigure::Server<node::TutorialsConfig>::CallbackType f;
    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

    image_transport::Subscriber image_sub_ = it_.subscribe("/camera/image_color", 1,&imageCb);

    // Publisher lanepub = nh_.advertise<geometry_msgs::Polygon>("lane_points", 1);
    Publisher lanes_pub = nh_.advertise<sensor_msgs::LaserScan>("/lanes", 10);
    
    Rate r(1);
    while(ok())
    {
        if(flag==0)
        {
           cout<< "Waiting for Image" << endl;
        }
        else
        {          
            lanes_pub.publish(scan_global);
        }


		flag=0;
		waitKey(500);
		spinOnce();
		r.sleep();
    }

    destroyAllWindows();
}
