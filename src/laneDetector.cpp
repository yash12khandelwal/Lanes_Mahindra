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
//#include "armadillo"

using namespace std;
using namespace cv;

static const std::string OPENCV_WINDOW = "Image window";
int flag=1;
bool FIRST_FRAME = true;
arma::vec curve;

string image_name;

bool tracking_status = false;
int frames_tracked = 0;

bool last_frame_low_features = false, current_frame_low_features = false;
bool one_frame_low_features = false;

VideoWriter out;

int h, w, horizon, tranformedPoints0_lowerbound, tranformedPoints0_upperbound, point1_y, point2_x, horizon_offset;
float hysterisThreshold_min, hysterisThreshold_max, variance;

// Mat transform = (Mat_<double>(3, 3) << 4.85067873e+01,   2.62964706e+02,  -4.48763439e+04, 3.87871256e-15,   4.00615385e+02,  -4.84907692e+04, 6.25698360e-18,   3.45701357e-01,  1.00000000e+00);

// Mat h = (Mat_<double>(3, 3) << 1, 0, 1, 0, 0, 1, 1, 1,  0);

// Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1;

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

    Mat edgeFeatures(1000, 800 ,CV_8UC1, Scalar(0));
    for( size_t i = 0; i < lane_lines_top.size(); i++ )
    {
        Vec4i l = lane_lines_top[i];
        line(edgeFeatures, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,255), 1, CV_AA);
    }

    if(debug==true)
    {
        Mat filtered_lines = img.clone();
        for( size_t i = 0; i < lane_lines.size(); i++ )
        {
            Vec4i l = lane_lines[i];
            line(filtered_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0, 0), 5, CV_AA);
        }

        namedWindow("edges", WINDOW_NORMAL);
        namedWindow("edgeFeatures", WINDOW_NORMAL);
        namedWindow("filtered_lines", WINDOW_NORMAL);

        imshow("edges", edges);
        imshow("edgeFeatures", edgeFeatures);
        imshow("filtered_lines", filtered_lines);
    }

    // cvtColor(edgeFeatures, edgeFeatures, CV_BGR2GRAY);
    return edgeFeatures;
}

// Mat findRoadBoundaries(Mat img, bool debug)
// {
//     Mat road = imread("/home/tejus/Documents/ml/KittiSeg/results/"+image_name);
//     Mat road_top, road_boundary;
//     Mat transform = (Mat_<double>(3, 3) << 4.85067873e+01,   2.62964706e+02,  -4.48763439e+04, 3.87871256e-15,   4.00615385e+02,  -4.84907692e+04, 6.25698360e-18,   3.45701357e-01,  1.00000000e+00);

//     warpPerspective(road, road_top, transform, Size(800, 1000), INTER_NEAREST, BORDER_CONSTANT);

//     Canny(road_top, road_boundary, 200, 300);

//     if(debug)
//     {
//         imshow("road_top", road_top);
//         imshow("road_boundary", road_boundary);
//     }

//     return road_boundary;
// }


Mat findIntensityMaxima(Mat img, bool debug)
{
    Mat topview;
    Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
    
    warpPerspective(img, topview, transform, Size(800, 1000), INTER_NEAREST, BORDER_CONSTANT);

    // topview = topview(Rect(100,0,500,900));
    GaussianBlur(topview, topview, Size(5, 15), 0, 0);
    blur(topview, topview, Size(25,25));

    namedWindow("topview", WINDOW_NORMAL);
    imshow("topview", topview);
    
    // if (debug==true)
    // {
    //     namedWindow("topview", WINDOW_NORMAL);
    //     imshow("topview", topview);
    // }

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
    // imshow("template1", getTemplateX2(2.1, h, w, -10));
    // imshow("template2", getTemplateX2(2.1, h, w, 0));
    // imshow("template3", getTemplateX2(2.1, h, w, 10));
    // imshow("template4", getTemplateX2(2.1, h, w, 20));
    // imshow("template5", getTemplateX2(2.1, h, w, 30));
    // imshow("template6", getTemplateX2(2.1, h, w, -20));
    // imshow("template7", getTemplateX2(2.1, h, w, -30));

    Mat t = t0-t0;
    for(int i=0;i<t.rows;i++)
        for(int j=0;j<t.cols;j++)
        {
            t.at<float>(i, j) = max( t4.at<float>(i, j), max(t3.at<float>(i, j), max(t0.at<float>(i, j), max(t1.at<float>(i, j), t2.at<float>(i, j)))));
            t.at<float>(i, j) = max( t.at<float>(i, j), max(t5.at<float>(i, j), t6.at<float>(i, j)) );
        }

    namedWindow("t", WINDOW_NORMAL);
    imshow("t", t);
    hysterisThreshold(t, topview, hysterisThreshold_min, hysterisThreshold_max);

    Mat result=Mat(topview.rows+h-1, topview.cols+w-1,CV_8UC3, Scalar(0,0,0));
    for(int i=0;i<topview.rows;i++)
        for(int j=0;j<topview.cols;j++)
            result.at<Vec3b>(i+(h-1)/2,j+(w-1)/2)={255*topview.at<float>(i,j), 255*topview.at<float>(i,j), 255*topview.at<float>(i,j)};


    if(debug==true)
    {
        namedWindow("intensityMaxima", WINDOW_NORMAL);
        imshow("intensityMaxima", result);
    }

    // waitKey(0);

    return result;
}

void initializeCurve()
{
    cout<<"Reinitializing !!"<<endl<<endl;
    curve << 0 << 0 << 380 << 300;
}


bool checkLaneChange(Mat img)
{
    Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
    

    transform=transform.inv();
    int x = 0 ;
    int y_left = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0) - curve[3]/2;
    int y_right = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0) + curve[3]/2;

    x = 800;
    int transformed_yleft = (y_left*transform.at<double>(0, 0)+x*transform.at<double>(0, 1)+1.0*transform.at<double>(0, 2))/(y_left*transform.at<double>(2, 0)+x*transform.at<double>(2, 1)+1.0*transform.at<double>(2, 2));    
    int transformed_yright = (y_right*transform.at<double>(0, 0)+x*transform.at<double>(0, 1)+1.0*transform.at<double>(0, 2))/(y_right*transform.at<double>(2, 0)+x*transform.at<double>(2, 1)+1.0*transform.at<double>(2, 2));

    if(transformed_yright<(img.cols/2))
    {
        cout<<"Lane shift right"<<endl;
        curve[2] += curve[3];
        return true;        
    }

    else if(transformed_yleft>(img.cols/2))
    {
        cout<<"Lane shift left"<<endl;
        curve[2] -= curve[3];
        return true;
    }

    return false;
}

bool checkPrimaryLane(Mat boundary, Mat edges, Mat intensityMaxima)
{
    // making a feature image similar to size of edges and initilalizing it with 0
    Mat features = edges - edges;
    
    // editing features on the basis of intesityMaxima, boundary, edges image
    // making particular pixel value of feature image 1 on basis of condition
    
    cout << boundary.rows << " " << boundary.cols << endl;
    cout << intensityMaxima.rows << " " << intensityMaxima.cols << endl;
    cout << edges.rows << " " << edges.cols << endl;

    for(int i=0;i<features.rows;i++)
        for(int j=0;j<features.cols;j++)
        {
            if(intensityMaxima.at<Vec3b>(i, j)[0]>0 || boundary.at<uchar>(i, j)>0 || edges.at<uchar>(i, j)>0)
                features.at<uchar>(i, j) = 255;
        }

    // namedWindow("features1", WINDOW_NORMAL);
    // imshow("features1", features);

    float error = 0;
    int n_points = 0;
    int nl_points=0, nr_points=0;

    cout<<"check"<<endl;

    for(int i=150;i<edges.rows;i++)
    {
        float w = curve[3];
        int k1 = 25;
        int k2 = 15;
        int x = edges.rows - i;
        int y = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0);

        float min_row_error = 99999;

        for(int offset = w/2 - k1; offset < w/2 + k2; offset++)
        {
            for(int l = 0; l < 2; l++ )
            {
                if( l > 0 )
                    offset = -offset;

                int j = (int)(y+offset);

                if(j>70 && j<600)
                {
                    if(features.at<uchar>(i, j)>0)
                    {
                        min_row_error = min(min_row_error, (float)pow(j-y, 2));
                        n_points++;

                        if(l==0)
                            nr_points++;
                        else
                            nl_points++;
                    }
                }
                if( l > 0 )
                    offset = -offset;
            }
        }
        error += min_row_error;
    }

    cout<<"error: " << error<<" , "<< "error/(n_points+1): " << error/(n_points+1) <<" , "<< "n_points: " << n_points << endl;
    cout<< "nl_points: " << nl_points <<","<< "nr_points: " << nr_points << endl;
    cout<<"max: " << max(nl_points, nr_points) << endl;
    cout<<"min: " << min(nl_points, nr_points) << endl;

    last_frame_low_features = current_frame_low_features;
    //if the minimum is very low or max
    cout<< "first: "<< (max(nl_points, nr_points) < 1000 && min(nl_points, nr_points) < 200) << endl;
    cout<< "second: "<< (min(nl_points, nr_points) < 100) << endl;


    current_frame_low_features = (max(nl_points, nr_points) < 1000  && min(nl_points, nr_points) < 200) || min(nl_points, nr_points) < 100;

    one_frame_low_features = min(nl_points, nr_points) < 250;
    
    cout<< "tracking status: " << tracking_status << endl;
    cout<< "current_frame_low_features: " << current_frame_low_features << endl;
    cout<< "last_frame_low_features: " << last_frame_low_features << endl;

    if(tracking_status==true)
    {
        if(current_frame_low_features && last_frame_low_features)
        {
            frames_tracked = 0;
            tracking_status = false;
            return false;
        }
        return true;
    }
    else
    {
        if(current_frame_low_features || last_frame_low_features)
        {
            frames_tracked = 0;
            return false;
        }
        else
        {
            frames_tracked++;

            if(frames_tracked>=5)
            {
                tracking_status = true;
                return true;
            }
            return false;
        }
    }
}

vector<Point> fitCurve(Mat img, Mat boundary, Mat edges, Mat intensityMaxima, bool debug=false)
{
    if(FIRST_FRAME==true)
    {
        //initializeCurve();
        curve << 0 << 0 << 380 << 320;
        FIRST_FRAME = false;
    }

    tracking_status = checkPrimaryLane(boundary, edges, intensityMaxima);
    
    if(!tracking_status)
    {
        initializeCurve();
    }

    checkLaneChange(img);

    float w = curve[3];
    int k1 = 15;
    int k2 = 10;

    int n_points = 0;
    for(int i=150;i<edges.rows;i++)
    {
        int x = edges.rows - i;
        int y = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0);

        bool found_closest_left = false;
        bool found_closest_right = false;
        for(int offset = w/2 - k1; offset < w/2 + k2; offset++)
        {
            for(int l = 0; l < 2; l++ )
            {
                if( l > 0 )
                    offset = -offset;

                int j = (int)(y+offset);

                if(j>70 && j<600)
                {
                    if(intensityMaxima.at<Vec3b>(i, j)[0]>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) ))
                    {
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else if(edges.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) ))
                    {
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else if(boundary.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) ))
                    {
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                }
                if( l > 0 )
                    offset = -offset;
            }
        }
    }

    cout<< "--------------------------" <<n_points<<endl;
    // if(n_points==0)
    // {
    //  cout<<"Tracking lost: No points on curve!"<<endl;
    //  return;
    // }
    // else
    // {
    //  cout<<"n_points: "<<n_points<<endl;
    // }

    // n_points = 10000;
    arma::mat B(n_points, 4);
    arma::vec X(n_points);
    n_points = 0;

    //Mat all_features = edges - edges;
    Mat all_features=Mat(800, 1000,CV_8UC3, Scalar(0,0,0));
    for(int i=0;i<edges.rows;i++)
        for(int j=0;j<edges.cols;j++)
        {
            if(intensityMaxima.at<Vec3b>(i, j)[0]>5)
            {
                all_features.at<Vec3b>(i, j) = {255, 100, 100};
            }
            else if(edges.at<uchar>(i, j)>5)
            {
                all_features.at<Vec3b>(i, j) = {100, 255, 100};
            }
            else if(boundary.at<uchar>(i, j)>5)
            {
                all_features.at<Vec3b>(i, j) = {100, 100, 255};
            }
        }
    //Mat features = intensityMaxima - intensityMaxima;
    Mat features=Mat(800, 1000,CV_8UC3, Scalar(0,0,0));

    for(int i=150;i<edges.rows;i++)
    {
        int x = edges.rows - i;
        int y = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0);

        bool found_closest_left = false;
        bool found_closest_right = false;
        for(int offset = w/2 - k1; offset < w/2 + k2; offset++)
        {
            for(int l = 0; l < 2; l++ )
            {
                if( l > 0 )
                    offset = -offset;

                int j = (int)(y+offset);

                if(j>70 && j<600)
                {
                    if(intensityMaxima.at<Vec3b>(i, j)[0]>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) ))
                    {
                        features.at<Vec3b>(i, j) = {255, 100, 100};
                        arma::rowvec temp;
                        temp << x*x << x << 1 << (-l + 0.5);
                        B.row(n_points) = temp;
                        X[n_points] = j;
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else if(edges.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) ))
                    {
                        features.at<Vec3b>(i, j) = {100, 255, 100};
                        arma::rowvec temp;
                        temp << x*x << x << 1 << (-l + 0.5);
                        B.row(n_points) = temp;
                        X[n_points] = j;
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else if(boundary.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) ))
                    {
                        features.at<Vec3b>(i, j) = {100, 100, 255};
                        arma::rowvec temp;
                        temp << x*x << x << 1 << (-l + 0.5);
                        B.row(n_points) = temp;
                        X[n_points] = j;
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else
                        features.at<Vec3b>(i, j) = {100, 100, 100};
                }
                if( l > 0 )
                    offset = -offset;
            }
        }
    }

    double lambda = 0.1; 
    arma::vec offset;
    offset << 0 << 0 << 380 << 300;
    arma::vec new_curve = inv(B.t()*B + lambda*arma::eye(4, 4))*(B.t()*X + lambda*offset);

    cout<<"new curve: "<<new_curve<<endl;

    for(int i=0;i<n_points;i++)
    {
        int x = B.at(i, 1);
        int y = X[i];
        features.at<Vec3b>(features.rows-x, y) = {255, 255, 255};
    }

    bool large_width_change = abs((curve[3]-new_curve[3])/curve[3]) > 0.5 || new_curve[3] < 250 || new_curve[3] > 400;
    if(!large_width_change && !one_frame_low_features)
    {
        curve[0] = (curve[0]+new_curve[0])/2;
        curve[1] = (curve[1]+new_curve[1])/2;
        curve[2] = (curve[2]+new_curve[2])/2;
        curve[3] = (curve[3]+new_curve[3])/2;
        cout << "width updated |||||||||||||||||||||||||||||||"<< endl;
    }
    
    else
    {
        curve[0] = (curve[0]+new_curve[0])/2;
        curve[1] = (curve[1]+new_curve[1])/2;
        curve[2] = new_curve[2] - new_curve[3]/2 + curve[3]/2;
        curve[3] = curve[3];
        cout<<"width not updated!..................................."<<endl;
    }
    
    if(debug==true)
    {
        namedWindow("features", WINDOW_NORMAL);
        namedWindow("all_features", WINDOW_NORMAL);

        imshow("features", features);
        imshow("all_features", all_features);
    }

    Mat topview, lanes;
    Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);
    

    warpPerspective(img, topview, transform, Size(800, 1000), INTER_NEAREST, BORDER_CONSTANT);

    int y1, y2, y3, y4;
    for(int i=150;i<topview.rows;i++)
    {
        int x = topview.rows - i;
        int y = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0);


        if(y<70 || y>600) continue;

        topview.at<Vec3b>(i, y) = {0, 0, 255};
        topview.at<Vec3b>(i, y - w/2) = {255, 0, 0};
        topview.at<Vec3b>(i, y + w/2) = {255, 0, 0};

        topview.at<Vec3b>(i, y+1) = {0, 0, 255};
        topview.at<Vec3b>(i, y - w/2+1) = {255, 0, 0};
        topview.at<Vec3b>(i, y + w/2+1) = {255, 0, 0};

        topview.at<Vec3b>(i, y-1) = {0, 0, 255};
        topview.at<Vec3b>(i, y - w/2-1) = {255, 0, 0};
        topview.at<Vec3b>(i, y + w/2-1) = {255, 0, 0};
        if(i==300)
        {
            y1=y - w/2;
            y4=y + w/2;
        }
        if(i==999)
        {
            y2=y - w/2;
            y3=y + w/2;
        }
    }

    // float y1, y2;

    // y1 = curve[0]*pow(topview.rows-160, 2) + curve[1]*pow(topview.rows-160, 1) + curve[2]*pow(topview.rows-160, 0);
    // y2 = curve[0]*pow(1, 2) + curve[1]*pow(1, 1) + curve[2]*pow(1, 0);

        // temp.push_back(Point(y1 - w/2, 160));
    // temp.push_back(Point(y1 + w/2, 160));
    // temp.push_back(Point(y2 - w/2, topview.rows));
    // temp.push_back(Point(y2 + w/2, topview.rows));

    // cout<<"(x1,y1): "<< y1 - w/2 << 160 << endl;
    // cout<<"(x2,y2): "<< y1 + w/2 << 160 << endl;
    // cout<<"(x3,y3): "<< y2 - w/2 << topview.rows << endl;
    // cout<<"(x4,y4): "<< y2 + w/2 << topview.rows << endl;
  
    // imshow("trial",topview);
    warpPerspective(topview, lanes, transform.inv(), Size(1920, 1200), INTER_NEAREST, BORDER_CONSTANT);
    
    // those pixels in which lane was not tracked were made 0, so now making those pixel values same as orignal image pixel value
    for(int i=0;i<lanes.rows;i++)
        for(int j=0;j<lanes.cols;j++)
            if(lanes.at<Vec3b>(i,j)[0]==0 && lanes.at<Vec3b>(i,j)[1]==0 && lanes.at<Vec3b>(i,j)[2]==0)
            {
                lanes.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j)[0];
                lanes.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i,j)[1];
                lanes.at<Vec3b>(i,j)[2] = img.at<Vec3b>(i,j)[2];
            }

    if(!tracking_status)
        lanes = img;

    if(tracking_status)
        putText(lanes, "Tracking Lanes", Point(50,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 2);
    else
        putText(lanes, "Initializing Tracking.", Point(50,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);


    namedWindow("lanes", WINDOW_NORMAL);
    imshow("lanes", lanes);

    vector<Point> temp;
    temp.push_back(Point(800,y1));
    temp.push_back(Point(1000,y2));
    temp.push_back(Point(1000,y3));
    temp.push_back(Point(800,y4));
    return temp;
    
}

vector<Point> detect_lanes(Mat img, bool debug = true)
{
    // display orignal image
    if(debug==true)
    {
        namedWindow("original", WINDOW_NORMAL);
        imshow("original", img);
    }

    // Mat boundary = findRoadBoundaries(img, true);
    
    // initialize boundary with a matrix of (800*400)
    Mat boundary = Mat(1000, 800, CV_8UC1, Scalar(0));

    // intenity maximum image made
    Mat intensityMaxima = findIntensityMaxima(img, true);
    
    // image with edge features made
    Mat edgeFeature = findEdgeFeatures(img, false, true);

    // curve fit on the basis of orignal image, maxima intensity image and edgeFeature image
    
    vector<Point> v;
    v = fitCurve(img, boundary, edgeFeature, intensityMaxima, true);
    
    // printing curve dimensions
    // cout<<curve<<endl;
    return v;

}

vector<Point> a;

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

    a = detect_lanes(img,true);
    waitKey(1);
}

int main(int argc, char **argv)
{
    // ros declaration
    ros::init(argc, argv, "lanes");

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_(nh_);
    image_transport::Subscriber image_sub_;

    string h_s, w_s, hysterisThreshold_min_s, hysterisThreshold_max_s, horizon_s, tranformedPoints0_lowerbound_s, tranformedPoints0_upperbound_s, point1_y_s, point2_x_s, horizon_offset_s, variance_s;

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

    ros::Publisher lanepub;
    cv_bridge::CvImagePtr cv_ptr;
   
    geometry_msgs::Point32 p1;
    geometry_msgs::Point32 p2;
    geometry_msgs::Point32 p3;
    geometry_msgs::Point32 p4;

    image_sub_ = it_.subscribe("/camera/image_color", 1,&imageCb);

    lanepub = nh_.advertise<geometry_msgs::Polygon>("lane_points", 1);
    
    ros::Rate r(1);
    while(ros::ok())
    {
        if(a.empty()||flag==0)
        {
           cout<< "Waiting for Image" << endl;
        }
        else
        {          
            geometry_msgs::Polygon lanes;
            p1.x = a[0].x;
            p1.y = a[0].y;

            p2.x = a[1].x;
            p2.y = a[1].y;

            p3.x = a[2].x;
            p3.y = a[2].y;

            p4.x = a[3].x;
            p4.y = a[3].y;
            cout<< "publishing lanes"<<endl;
            lanes.points.push_back(p1);
            lanes.points.push_back(p2);
            lanes.points.push_back(p3);
            lanes.points.push_back(p4);

            lanepub.publish(lanes);
        }


       flag=0;

        ros::spinOnce();
        r.sleep();
    }
    waitKey(1);
    destroyAllWindows();
}
