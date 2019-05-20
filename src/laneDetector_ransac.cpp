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
#include "lsd.cpp"
#include "laneDetector_utils.cpp"
#include "../include/houghP.hpp"
#include "../include/laneDetector_utils.hpp"
#include "../include/lsd.h"
#include <string>
#include "armadillo"
#include "../include/ransac.hpp"
// #include "../include/mlesac.hpp"
#include <nav_msgs/Odometry.h>
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include <sensor_msgs/LaserScan.h>

#define yshift 0.33   /// distance from first view point to lidar in metres
#define angleshift 0.0524    /// angle between camera and lidar axis in radians    
#define bins 1080  /// no of bins

using namespace std;
using namespace cv;

static const std::string OPENCV_WINDOW = "Image window";

int flag=1;

arma::vec curve;
string image_name;

int h, w, horizon, tranformedPoints0_lowerbound, tranformedPoints0_upperbound, point1_y, point2_x, horizon_offset;
float hysterisThreshold_min, hysterisThreshold_max, variance, pixelPerMeter, cameraClearance;
model lanes, previous;
ofstream file;
vector<Point> cost_map_lanes;

sensor_msgs::LaserScan scan_global;

Mat findEdgeFeatures(Mat img, bool top_edges, bool debug);
Mat findIntensityMaxima(Mat img, bool debug);
vector<Point> detect_lanes(Mat img, bool debug);
void imageCb(const sensor_msgs::ImageConstPtr& msg);
void imageCb(const sensor_msgs::ImageConstPtr& msg);
sensor_msgs::LaserScan imageConvert(Mat image);

Mat findEdgeFeatures(Mat img, bool top_edges, bool debug)
{
    Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
        
    //medianBlur(img, img, 3);
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
        Mat topview;
        warpPerspective(img, topview, transform, Size(800, 1000), INTER_NEAREST, BORDER_CONSTANT);
        Canny(topview, edges, 200, 300);
        HoughLinesP(edges, lines_top, 1, CV_PI/180, 60, 60, 50);
        transformLines(lines_top, lines, transform.inv());
    }

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        Point2f inputPoints[2], transformedPoints[2];

        inputPoints[0] = Point2f( l[0], l[1] );
        inputPoints[1] = Point2f( l[2], l[3] ); 
        transformPoints(inputPoints, transformedPoints, transform, 2);

        if(transformedPoints[0].x<tranformedPoints0_lowerbound || transformedPoints[0].x>tranformedPoints0_upperbound || transformedPoints[1].x<tranformedPoints0_lowerbound || transformedPoints[1].x>tranformedPoints0_upperbound || l[1] < point1_y || l[2] < point2_x)
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

    transformLines(lane_lines, lane_lines_top, transform);
    Mat edgeFeatures;
    Mat filtered_lines = Mat(img.size(),CV_8UC1, Scalar(0));
    for( size_t i = 0; i < lane_lines.size(); i++ )
    {
        Vec4i l = lane_lines[i];
        line(filtered_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 5, CV_AA);
    }
    warpPerspective(filtered_lines, edgeFeatures, transform, Size(800,1000));

    if(debug==true)
    {

        // namedWindow("edges", WINDOW_NORMAL);
        // namedWindow("edgeFeatures", WINDOW_NORMAL);
        // namedWindow("filtered_lines", WINDOW_NORMAL);

        // //imshow("edges", edges);
        // //imshow("edgeFeatures", edgeFeatures);
        // //imshow("filtered_lines", filtered_lines);
    }

    // cvtColor(edgeFeatures, edgeFeatures, CV_BGR2GRAY);
    return edgeFeatures;
}

Mat findIntensityMaxima(Mat img, bool debug)
{
    Mat topview;
    Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
    
    warpPerspective(img, topview, transform, Size(800, 1000), INTER_NEAREST, BORDER_CONSTANT);

    // topview = topview(Rect(100,0,500,900));
    GaussianBlur(topview, topview, Size(5, 15), 0, 0);
    blur(topview, topview, Size(25,25));

    // namedWindow("topview", WINDOW_NORMAL);
    // //imshow("topview", topview);

    cvtColor(topview, topview, CV_BGR2GRAY);

    
    medianBlur(topview, topview, 3);
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

    hysterisThreshold(t, topview, hysterisThreshold_min, hysterisThreshold_max);

    Mat result=Mat(topview.rows+h-1, topview.cols+w-1,CV_8UC1, Scalar(0));
    for(int i=0;i<topview.rows;i++)
        for(int j=0;j<topview.cols;j++)
            result.at<uchar>(i+(h-1)/2,j+(w-1)/2)=255*topview.at<float>(i,j);


    if(debug==true)
    {
        // namedWindow("intensityMaxima", WINDOW_NORMAL);
        // //imshow("intensityMaxima", result);
    }

    return result;
}


vector<Point> detect_lanes(Mat img, bool debug = true)
{
    Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
    
    // display orignal image
    if(debug==true)
    {
        // namedWindow("original", WINDOW_NORMAL);
        // //imshow("original", img);
    }

    // Mat boundary = findRoadBoundaries(img, true);
    
    // initialize boundary with a matrix of (800*400)
    Mat boundary = Mat(1000, 800, CV_8UC1, Scalar(0));

    // intenity maximum image made
    Mat intensityMaxima = findIntensityMaxima(img, true);
    
    // image with edge features made
    Mat edgeFeature = findEdgeFeatures(img, false, true);

    // curve fit on the basis of orignal image, maxima intensity image and edgeFeature image

    Mat all_features = Mat(1000, 800,CV_8UC1, Scalar(0));

    int y = 400;
    int w = 320;
    int k1 = 20;
    int k2 = 170;

    medianBlur(intensityMaxima,intensityMaxima, 5);

    for(int i=0; i<edgeFeature.rows; i++)
    {
        for(int l = 0; l < 2; l++ )
        {
            for(int offset = w/2 - k1; offset < w/2 + k2; offset++)
            {
                if( l > 0 )
                    offset = -offset;
                
                int j = (int)(y+offset);
                // cout <<"i: "<<i<<"j: "<< j<<  endl;
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
    // namedWindow("all_features", WINDOW_NORMAL);
    // //imshow("all_features", all_features);


    Mat lanes_by_ransac = Mat(1000, 800, CV_8UC3, Scalar(0,0,0));
    lanes = getRansacModel(all_features, previous);
    previous=lanes;

    if(lanes.a1==0 && lanes.b1==0 && lanes.c1==0 && lanes.a2!=0 && lanes.b2!=0 && lanes.c2!=0)
    {
        for(int i=0;i<lanes_by_ransac.rows;i++)
        {
            for(int j=0;j<lanes_by_ransac.cols;j++)
            {
                if(fabs(lanes.a2*i*i+lanes.b2*i+lanes.c2-j)<3)
                {
                    lanes_by_ransac.at<Vec3b>(i,j)[0]=255;
                    lanes_by_ransac.at<Vec3b>(i,j)[1]=255;
                    lanes_by_ransac.at<Vec3b>(i,j)[2]=255;
                }
            }
        }
    }

    else if(lanes.a1!=0 && lanes.b1!=0 && lanes.c1!=0 && lanes.a2==0 && lanes.b2==0 && lanes.c2==0)
    {
        for(int i=0;i<lanes_by_ransac.rows;i++)
        {
            for(int j=0;j<lanes_by_ransac.cols;j++)
            {
                if(fabs(lanes.a1*i*i+lanes.b1*i+lanes.c1-j)<3) 
                    {
                        lanes_by_ransac.at<Vec3b>(i,j)[0]=255;
                        lanes_by_ransac.at<Vec3b>(i,j)[1]=255;
                        lanes_by_ransac.at<Vec3b>(i,j)[2]=255;
                    }
            }
        }
    }
    
    else
    {
        for(int i=0;i<lanes_by_ransac.rows;i++)
        {
            for(int j=0;j<lanes_by_ransac.cols;j++)
            {
                if(fabs(lanes.a1*i*i+lanes.b1*i+lanes.c1-j)<3) 
                    {
                        lanes_by_ransac.at<Vec3b>(i,j)[0]=255;
                        lanes_by_ransac.at<Vec3b>(i,j)[1]=255;
                        lanes_by_ransac.at<Vec3b>(i,j)[2]=255;
                    }
            }
        }
        for(int i=0;i<lanes_by_ransac.rows;i++)
        {
            for(int j=0;j<lanes_by_ransac.cols;j++)
            {
                if(fabs(lanes.a2*i*i+lanes.b2*i+lanes.c2-j)<3)
                {
                    lanes_by_ransac.at<Vec3b>(i,j)[0]=255;
                    lanes_by_ransac.at<Vec3b>(i,j)[1]=255;
                    lanes_by_ransac.at<Vec3b>(i,j)[2]=255;
                }
            }
        }
    }

    namedWindow("lanes_by_ransac", WINDOW_NORMAL);
    imshow("lanes_by_ransac", lanes_by_ransac);

    Mat frontview;
    warpPerspective(lanes_by_ransac, frontview, transform.inv(), Size(1920, 1200), INTER_NEAREST, BORDER_CONSTANT);
    
    vector<Point> v;
    float a;
    float b;

    for(int i = 0; i < frontview.rows ; i += 2)
    {
        for(int j = 0; j < frontview.cols ; j += 2)
        {
            if(frontview.at<uchar>(i,j) == 255)
                a = cameraClearance + (frontview.rows - i)*pixelPerMeter;
                b = ((frontview.cols/2.0) - j)*pixelPerMeter;
                v.push_back(Point(a,b));
        }
    }

    namedWindow("checking", WINDOW_NORMAL);
    imshow("checking", lanes_by_ransac);

    scan_global = imageConvert(lanes_by_ransac);

    // for(int i=0; i<img.rows;i++)
    // {
    //     for(int j=0; j< img.cols; j++)
    //     {
    //         if(frontview.at<Vec3b>(i,j)[0]>235) 
    //             {
    //                 img.at<Vec3b>(i,j)[0]=0;
    //                 img.at<Vec3b>(i,j)[1]=0;
    //                 img.at<Vec3b>(i,j)[2]=255;
    //             }
    //     }
    // }
    // namedWindow("lanes1", WINDOW_NORMAL);
    //imshow("lanes1", img);
    
    return v;
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
        cout << img.rows << img.cols << endl;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if( !img.data ) { printf("Error loading A \n"); return ; }

    cost_map_lanes = detect_lanes(img,true);
    // waitKey(0);
}

sensor_msgs::LaserScan imageConvert(Mat img)    /// Input binary image for conversion to laserscan
{
	namedWindow("check", WINDOW_NORMAL);
    imshow("check", img);

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
				float a = (j - col/2)/pixelPerMeter;
				float b = (row - i)/pixelPerMeter + yshift;

				double angle = atan(a/b);

				double r = sqrt(a*a  + b*b);

				int k = (angle - scan.angle_min)/(scan.angle_increment);
				scan.ranges[bins-k-1] = r ;
			}
		}
	}

	return scan;    /// returns Laserscan data
}
/*
sensor_msgs::LaserScan imageConvert(Mat image) 
{  
    sensor_msgs::LaserScan scan;
    scan.angle_min = -2.36;
    scan.angle_max = 2.36;
    scan.angle_increment = 0.004;
    scan.header.stamp = ros::Time::now();
    scan.header.frame_id = "laser";
    scan.range_min = 0;
    scan.range_max = 30;
    scan.time_increment = (float)(0.025/1181);
    scan.scan_time = 0.025;
    scan.ranges.resize(1181);
    
    for(int t=0;t<=1180;t++)
    {
        scan.ranges[t] = 1000;  //maximum range value
    }
    
    for(int i=0;i<image.rows;i++)
    {    
        for(int j=0;j<image.cols;j++)
        {
            float angle,dist;
            float x,y;

            if(image.at<uchar>(i,j)>127)
            {
                // x = (img.rows-1- i + cameraClearance*pixelPerMeter)/pixelPerMeter;
                x = (float)((image.rows-i)/(pixelPerMeter)) + cameraClearance;
                y = (float)((image.cols/2.0) - j)/(pixelPerMeter);
            file << x << "," << y<< endl;

                if (x != 0) {
                    angle = -(1.0)*atan2(y,x);
                }
                else 
                { 
                  if(y>0) angle=-1.57;
                  else angle=(1.57);
                }


                // if(x!=0)
                // {
                //     angle = (1.0)*atan2((1.0)*y,x);
                // }


                // else 
                // { 
                //     if(y>0)
                //         angle = CV_PI/2.0;

                //     else 
                //         angle = (-1)*(CV_PI/2.0);
                // }

                dist=sqrt(x*x+y*y);
                int index=(int)(fabs(angle-scan.angle_min)/scan.angle_increment);

                // cout <<"x: " << x << " y: " << y << " angle: " << angle << " dist: " << dist << endl;
                // int index= (int)(fabs(scan.angle_min - angle)/scan.angle_increment);

                // cout << "index: " << index << endl;
                if(scan.ranges[index]>dist)
                    scan.ranges[index] = dist;
                }
        }
    }
    int a;
    cin>>a;
    return scan;
}
*/

// sensor_msgs::LaserScan imageConvert(Mat image) 
// {  
//     sensor_msgs::LaserScan scan;
//     scan.angle_min = 0;
//     scan.angle_max = -1;
//     scan.angle_increment = 0.004;
//     scan.header.stamp = ros::Time::now();
//     scan.header.frame_id = "laser";
//     scan.range_min=0;
//     scan.range_max=30;
//     int num_points = 1180;
//     scan.time_increment=(float)(0.025/1181);
//     scan.scan_time=0.025;
//     scan.ranges.resize(1181);

//     int centre = image.cols/2 + .3*pixelPerMeter;
    
//     for(int t=0;t<=num_points;t++)
//     {
//         scan.ranges[t] = 10;  //maximum range value
//     }

//     // namedWindow("Pixel_waala",WINDOW_NORMAL);
//     for(int i=0;i<image.rows;i++)
//     {
//        for(int j=0;j<image.cols;j++)
//         {
//           // Mat black(image.rows,image.cols,CV_8UC1,Scalar(0,0,0));
//           // black.at<uchar>(i,j) = 255;
//           // //imshow("Pixel_waala",black);
//           // waitKey(0);
//           // ROS_INFO("what");
//           float angle,dist;
//           float x,y;

//           if(image.at<uchar>(i,j)>128)
//           {

//             y=(float)(j-centre)/(1.0*pixelPerMeter);
//             x=(float)((image.rows-i)+ 0*pixelPerMeter)/pixelPerMeter;
            
//             file << x << "," << y<< endl;
            
//             if(y!=0)
//               angle = (-1.0)*atan((1.0)*x/y);
//             else 
//               { 
//                 if(x>0) angle=-1.57;
//                 else angle=(1.57);
//               }

//             dist=sqrt(x*x+y*y);

//             cout << "r : " << dist << ", theta : " << angle << endl;

//             int index=(int)((angle-scan.angle_min)/scan.angle_increment);

//             cout << index << endl;
//             cout << "----------" << endl;

//              if(scan.ranges[index]>dist)
//               scan.ranges[index]=dist;
//           }
//       }
//     }
    // int a;
    // // cin>>a;
    // return scan;
// }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lanes");

    ros::NodeHandle nh_;
    
    string h_s, w_s, hysterisThreshold_min_s, hysterisThreshold_max_s, horizon_s, tranformedPoints0_lowerbound_s, tranformedPoints0_upperbound_s, point1_y_s, point2_x_s, horizon_offset_s, variance_s, pixelPerMeter_s, cameraClearance_s;
    file.open("xy.txt");
    h = stod(nh_.param<std::string>("/h",h_s));    
    w = stod(nh_.param<std::string>("/w",w_s));
    hysterisThreshold_min = stod(nh_.param<std::string>("/hysterisThreshold_min",hysterisThreshold_min_s));
    hysterisThreshold_max = stod(nh_.param<std::string>("/hysterisThreshold_max",hysterisThreshold_max_s));
    horizon = stod(nh_.param<std::string>("/horizon",horizon_s));
    tranformedPoints0_lowerbound = stod(nh_.param<std::string>("/tranformedPoints0_lowerbound",tranformedPoints0_lowerbound_s));
    tranformedPoints0_upperbound = stod(nh_.param<std::string>("/tranformedPoints0_upperbound",tranformedPoints0_upperbound_s));
    point1_y =  stod(nh_.param<std::string>("/point1_y",point1_y_s));
    point2_x = stod(nh_.param<std::string>("/point2_x",point2_x_s));
    horizon_offset = stod(nh_.param<std::string>("/horizon_offset",horizon_offset_s));
    variance = stod(nh_.param<std::string>("/variance", variance_s));
    pixelPerMeter = stod(nh_.param<std::string>("/pixelPerMeter", pixelPerMeter_s));
    cameraClearance = stod(nh_.param<std::string>("/cameraClearance", cameraClearance_s));

    cv_bridge::CvImagePtr cv_ptr;
    image_transport::ImageTransport it_(nh_);
    image_transport::Subscriber image_sub_ = it_.subscribe("/camera/image_color", 1,&imageCb);
    ros::Publisher lanepub = nh_.advertise<geometry_msgs::Polygon>("lane_points", 1);
    ros::Publisher lanes_pub = nh_.advertise<sensor_msgs::LaserScan>("/lanes", 10);
    
    ros::Rate r(1);
    while(ros::ok())
    {
        if(cost_map_lanes.empty()||flag==0)
        {
           cout<< "Waiting for Image" << endl;
        }
        else
        {          
            lanes_pub.publish(scan_global);
        }


       flag=0;
       waitKey(30);
        ros::spinOnce();
        r.sleep();
    }
    // waitKey(0);
    destroyAllWindows();
}
