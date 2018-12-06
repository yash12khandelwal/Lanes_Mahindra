#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <iostream>

#include "../include/houghP.hpp"
#include "../include/laneDetector_utils.hpp"
#include "../include/lsd.h"

#include "armadillo"


using namespace std;
using namespace cv;

bool FIRST_FRAME = true;
arma::vec curve;

string image_name;

bool tracking_status = false;
int frames_tracked = 0;

bool last_frame_low_features = false, current_frame_low_features = false;
bool one_frame_low_features = false;

VideoWriter out;

//Evaluation----------------------------------------------------------
double error_histogram_right[1242];
double error_histogram_left[1242];
double error_histogram_mean[1242];
string label_name;

ofstream eval_file;

// int evaluate(Mat img,bool debug,int frame){

// 	label_name = string(10-int_to_string(frame).length(),'0')+int_to_string(frame)+".png";
// 	Mat label_img = imread("/home/tejus/Downloads/2011_09_26_drive_0032_sync/"+label_name);

// 	float num_points=0;
// 	float error=0;

// 	for(int i=0;i<label_img.rows;i++)
// 	{
// 		int x = 800 - i;
//         int y = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0);
//         float w = curve[3];

//         for(int j=0;j<label_img.cols;j++)
// 		{
// 			if(label_img.at<Vec3b>(i,j)[0]>200)
// 			{
// 				num_points++;
// 				error+= abs(y-w/2-j);
// 			}
// 			else if(label_img.at<Vec3b>(i,j)[1]>200)
// 			{
// 				num_points++;
// 				error+= abs(y+w/2-j);
// 			}
// 		}
// 	}

// 	if(num_points>=1)
// 		eval_file<<error/num_points<<endl;

// 	cout<<"Avg. error: "<<error/num_points<<"  num_points: "<<num_points<<endl<<endl;

// }

// void display_evaluation(int total_frame){

// 	Mat img(375,1242,CV_8UC3,Scalar(0,0,0));

//  	for(int i=0;i<1242;i++){
//  		error_histogram_mean[i]=(error_histogram_left[i]+error_histogram_right[i])/(2);
//  		cout<<error_histogram_mean[i]<<" ";
//  		for(int j=0;j<error_histogram_mean[i];j++){
//  			img.at<Vec3b>(375-j,i)[1]=255;

//  		}

// 	}
// 	imshow("Evaluation",img);
// }

Mat findEdgeFeatures(Mat img, bool top_edges, bool debug)
{
    //medianBlur(img, img, 3);
    Mat transform = (Mat_<double>(3, 3) << 3.57576055e-01,   2.07240514e+00,  -5.91721615e+02, 1.67087151e-01,   9.43860355e+00,  -2.02168615e+03, 1.81143049e-04,   1.03884056e-02,  -1.97849048e+00);
    int horizon = 180;              // Hardcode for kitti dataset
    vector<Vec4i> lines, lines_top;
    vector<int> line_lens;
    Mat edges;

    if(top_edges == false)
    {
        //Canny(img, edges, 200, 300);
       // HoughLinesP(edges, lines, 1, CV_PI/180, 60, 50, 10);
            cv::Mat src = img;
            cv::Mat tmp, src_gray;
            cv::cvtColor(src, tmp, CV_RGB2GRAY);
            tmp.convertTo(src_gray, CV_64FC1);

            int cols  = src_gray.cols;
            int rows = src_gray.rows;

            image_double image = new_image_double(cols, rows);
            image->data = src_gray.ptr<double>(0);
            ntuple_list ntl = lsd(image);

            cv::Mat lsd = cv::Mat::zeros(rows, cols, CV_8UC1);
            cv::Point pt1, pt2;
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

                // done by me
                if (width == 0)
                    continue;

                cv::line(lsd, pt1, pt2, cv::Scalar(255), width, CV_AA);
            }
            free_ntuple_list(ntl);
            edges=lsd;

    }
    else
    {
        Mat topview;
        warpPerspective(img, topview, transform, cv::Size(400, 800), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
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

        if(transformedPoints[0].x<75 || transformedPoints[0].x>325 || transformedPoints[1].x<75 || transformedPoints[1].x>325 || l[1] < 200 || l[2] < 200)
        {
            lines.erase(lines.begin() + i);
            //line_lens.erase(line_lens.begin() + i);
            i--;
        }
    }

    int is_lane[lines.size()];
    for(int i=0;i<lines.size();i++)
        is_lane[i]=0;

    for(size_t i=0;i<lines.size();i++)
        for(size_t j=0;j<lines.size();j++)
            if(abs(findIntersection(lines[i],lines[j])-horizon)<10)
            {
                //is_lane[i]+=line_lens[j];
                //is_lane[j]+=line_lens[i];
                is_lane[i]+=1;
                is_lane[j]+=1;
            }

    vector<Vec4i> lane_lines, lane_lines_top;

    for(int i=0;i<lines.size();i++)
        if(is_lane[i]>10)
            lane_lines.push_back(lines[i]);

    transformLines(lane_lines, lane_lines_top, transform);

    Mat edgeFeatures(800, 400 ,CV_8UC3, Scalar(0,0,0));
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
        imshow("edges", edges);
        imshow("edgeFeatures", edgeFeatures);
        imshow("filtered_lines", filtered_lines);
    }

    cvtColor(edgeFeatures, edgeFeatures, CV_BGR2GRAY);
    return edgeFeatures;
}

// Mat findRoadBoundaries(Mat img, bool debug)
// {
//     Mat road = imread("/home/tejus/Documents/ml/KittiSeg/results/"+image_name);
//     Mat road_top, road_boundary;

//     Mat transform = (Mat_<double>(3, 3) << 3.57576055e-01,   2.07240514e+00,  -5.91721615e+02, 1.67087151e-01,   9.43860355e+00,  -2.02168615e+03, 1.81143049e-04,   1.03884056e-02,  -1.97849048e+00);
//     warpPerspective(road, road_top, transform, cv::Size(400, 800), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

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
    Mat transform = (Mat_<double>(3, 3) << 3.57576055e-01,   2.07240514e+00,  -5.91721615e+02, 1.67087151e-01,   9.43860355e+00,  -2.02168615e+03, 1.81143049e-04,   1.03884056e-02,  -1.97849048e+00);
    warpPerspective(img, topview, transform, cv::Size(400, 800), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    if (debug==true)
        imshow("topview", topview);

    int h = 51, w = 35;
    cvtColor(topview, topview, CV_BGR2GRAY);
    GaussianBlur(topview, topview, Size(1, 11), 0.01, 4);
    Mat t0, t1, t2, t3, t4, t5, t6;
    matchTemplate(topview, getTemplateX2(2, h, w, -10), t0, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2, h, w,   0), t1, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2, h, w,  10), t2, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2, h, w,  -20), t3, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2, h, w,  +20), t4, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2, h, w,  +30), t5, CV_TM_CCOEFF_NORMED);
    matchTemplate(topview, getTemplateX2(2, h, w,  -30), t6, CV_TM_CCOEFF_NORMED);

    Mat t = t0-t0;
    for(int i=0;i<t.rows;i++)
        for(int j=0;j<t.cols;j++)
        {
            t.at<float>(i, j) = max( t4.at<float>(i, j), max(t3.at<float>(i, j), max(t0.at<float>(i, j), max(t1.at<float>(i, j), t2.at<float>(i, j)))));
            t.at<float>(i, j) = max( t.at<float>(i, j), max(t5.at<float>(i, j), t6.at<float>(i, j)) );
        }


    hysterisThreshold(t, topview, 0.2, 0.4);

    Mat result=Mat(topview.rows+h-1, topview.cols+w-1,CV_8UC3, Scalar(0,0,0));
    for(int i=0;i<topview.rows;i++)
        for(int j=0;j<topview.cols;j++)
            result.at<Vec3b>(i+(h-1)/2,j+(w-1)/2)={255*topview.at<float>(i,j), 255*topview.at<float>(i,j), 255*topview.at<float>(i,j)};


    if(debug==true)
    	imshow("intensityMaxima", result);

    return result;
}

void initializeCurve()
{
	cout<<"Reinitializing !!"<<endl<<endl;
    curve << 0 << 0 << 200 << 74;
}


bool checkLaneChange(Mat img){
    Mat transform = (Mat_<double>(3, 3) << 3.57576055e-01,   2.07240514e+00,  -5.91721615e+02, 1.67087151e-01,   9.43860355e+00,  -2.02168615e+03, 1.81143049e-04,   1.03884056e-02,  -1.97849048e+00);
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
    cout<<"c0"<<endl;
    Mat features = edges - edges;
    for(int i=0;i<features.rows;i++)
        for(int j=0;j<features.cols;j++)
        {
            if(intensityMaxima.at<Vec3b>(i, j)[0]>0 || boundary.at<uchar>(i, j)>0 || edges.at<uchar>(i, j)>0)
                features.at<uchar>(i, j) = 255;
        }

    float error = 0;
    int n_points = 0;
    int nl_points=0, nr_points=0;

    cout<<"check"<<endl;

    for(int i=350;i<edges.rows;i++)
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

                if(j>0 && j<400)
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
    cout<<error<<", "<<error/(n_points+1)<<", "<<n_points<<endl;
    cout<<nl_points<<","<<nr_points<<endl;
    cout<<max(nl_points, nr_points)<<endl;

    last_frame_low_features = current_frame_low_features;
    current_frame_low_features = (max(nl_points, nr_points) < 2500 && min(nl_points, nr_points) < 700) || min(nl_points, nr_points) < 500;

    one_frame_low_features = min(nl_points, nr_points) < 450;

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

void fitCurve(Mat img, Mat boundary, Mat edges, Mat intensityMaxima, bool debug=false)
{
    if(FIRST_FRAME==true)
    {
        //initializeCurve();
        curve << 0 << 0 << 200 << 60;
        //curve << 0.0015 << 0.3 << 216 << 155;
        //curve << 0 << 0.068 << 185.92 << 108.39;
        FIRST_FRAME = false;
    }

    checkPrimaryLane(boundary, edges, intensityMaxima);
    tracking_status = true;
    if(!tracking_status)
    {
        initializeCurve();
    }

    checkLaneChange(img);

    float w = curve[3];
    int k1 = 15;
    int k2 = 10;

    int n_points = 0;
    for(int i=350;i<edges.rows;i++)
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

                if(j>0 && j<400)
                {
                    if(intensityMaxima.at<Vec3b>(i, j)[0]>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) || true))
                    {
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else if(edges.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) || true))
                    {
                        n_points++;
                        if(l==0)
                            found_closest_right = true;
                        else
                            found_closest_left = true;
                    }
                    else if(boundary.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) || true))
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

    if(n_points==0)
    {
    	cout<<"Tracking lost: No points on curve!"<<endl;
    	return;
    }
    else
    {
    	cout<<"n_points: "<<n_points<<endl;
    }

    arma::mat B(n_points, 4);
    arma::vec X(n_points);
    n_points = 0;

    //Mat all_features = edges - edges;
    Mat all_features=Mat(800, 400,CV_8UC3, Scalar(0,0,0));
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
    Mat features=Mat(800, 400,CV_8UC3, Scalar(0,0,0));

    for(int i=350;i<edges.rows;i++)
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

                if(j>0 && j<400)
                {
                    if(intensityMaxima.at<Vec3b>(i, j)[0]>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) || true))
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
                    else if(edges.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) || true))
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
                    else if(boundary.at<uchar>(i, j)>5 && ((l==0 && !found_closest_right) || (l==1 && !found_closest_left) || true))
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
    //arma::vec new_curve = solve(B, X);
    //arma::vec new_curve = inv(B.t()*B)*(B.t()*X);

    double lambda = 0.1; 
    arma::vec offset;
    offset << 0 << 0 << 200 << 60;
    arma::vec new_curve = inv(B.t()*B + lambda*arma::eye(4, 4))*(B.t()*X + lambda*offset);

    // arma::vec new_curve = curve;

    // double beta1 = 0.9, beta2 = 0.995, learning_rate = 0.0001;

    // arma::vec first_moment, second_moment, second_moment_sqrt;
    // first_moment << 0 << 0 << 0 << 0;
    // second_moment << 0 << 0 << 0 << 0;
    // second_moment_sqrt = second_moment;

    // int j = 1;
    // while (j <= 10000){

    //     arma::mat step_error = (B*new_curve - X);
    //     arma::vec d_curve = B.t() * step_error * (1.0/B.n_rows);

    //     first_moment = beta1 * first_moment + (1-beta1) * d_curve;
    //     second_moment = beta2 * second_moment + (1-beta2) * (d_curve % d_curve);

    //     arma::vec first_unbias = first_moment / (1 - pow(beta1, j));
    //     arma::vec second_unbias = second_moment / (1 - pow(beta2, j));

    //     for(int i=0;i<second_moment.n_elem;i++)
    //     	second_moment_sqrt[i] = sqrt(second_unbias[i]);

    //     new_curve += -learning_rate * first_unbias / (second_moment_sqrt + 1e-7);

    //     j++;
    //     // if(j%20==0)
    //     // 	cout<<j<<": "<<norm(step_error, 2)/(2.0*B.n_rows)<<endl;
    // }

  


    cout<<"new curve: "<<new_curve<<endl;

    for(int i=0;i<n_points;i++)
    {
    	int x = B.at(i, 1);
        int y = X[i];
        features.at<Vec3b>(features.rows-x, y) = {255, 255, 255};
    }


    bool large_width_change = abs((curve[3]-new_curve[3])/curve[3]) > 0.5 || new_curve[3] < 40 || new_curve[3] > 150;
    if(!large_width_change && !one_frame_low_features)
    {
    	curve[0] = (curve[0]+new_curve[0])/2;
	    curve[1] = (curve[1]+new_curve[1])/2;
	    curve[2] = (curve[2]+new_curve[2])/2;
	    curve[3] = (curve[3]+new_curve[3])/2;
    }
    else
    {
    	curve[0] = (curve[0]+new_curve[0])/2;
	    curve[1] = (curve[1]+new_curve[1])/2;
	    curve[2] = new_curve[2] - new_curve[3]/2 + curve[3]/2;
	    curve[3] = curve[3];
   		cout<<"width not updated!"<<endl;
    }

    imshow("features", features);
    imshow("all_features", all_features);

    Mat topview, lanes;
    Mat transform = (Mat_<double>(3, 3) << 3.57576055e-01,   2.07240514e+00,  -5.91721615e+02, 1.67087151e-01,   9.43860355e+00,  -2.02168615e+03, 1.81143049e-04,   1.03884056e-02,  -1.97849048e+00);
    warpPerspective(img, topview, transform, cv::Size(400, 800), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    for(int i=160;i<topview.rows;i++)
    {
        int x = topview.rows - i;
        int y = curve[0]*pow(x, 2) + curve[1]*pow(x, 1) + curve[2]*pow(x, 0);

        if(y<0 || y>400)
        	continue;

        topview.at<Vec3b>(i, y) = {0, 0, 255};
        topview.at<Vec3b>(i, y - w/2) = {255, 0, 0};
        topview.at<Vec3b>(i, y + w/2) = {255, 0, 0};

        topview.at<Vec3b>(i, y+1) = {0, 0, 255};
        topview.at<Vec3b>(i, y - w/2+1) = {255, 0, 0};
        topview.at<Vec3b>(i, y + w/2+1) = {255, 0, 0};

        topview.at<Vec3b>(i, y-1) = {0, 0, 255};
        topview.at<Vec3b>(i, y - w/2-1) = {255, 0, 0};
        topview.at<Vec3b>(i, y + w/2-1) = {255, 0, 0};
    }
    warpPerspective(topview, lanes, transform.inv(), cv::Size(1242, 375), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
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

    imshow("lanes", lanes);
    cout<<"Writing: "<<tracking_status<<endl;
    out.write(lanes);
    return;
    
}

void detect_lanes(Mat img, bool debug = true)
{
    if(debug==true)
        imshow("original", img);

    //Mat boundary = findRoadBoundaries(img, true);
    Mat boundary = Mat(800, 400, CV_8UC1, Scalar(0));
    Mat intensityMaxima = findIntensityMaxima(img, true);
    Mat edgeFeature = findEdgeFeatures(img, false, true);



    fitCurve(img, boundary, edgeFeature, intensityMaxima, true);
    cout<<curve<<endl;
}

int main()
{
    //266=straight road
    //900=curved road
    out.open( "outputVideo1.avi", CV_FOURCC('D','I','V','X'), 10, cv::Size ( 1242, 375), true );
    // for(int i=850;i<=1100;i++)
    // {
    //     cout<<i<<endl;
    //     image_name = string(6-int_to_string(i).length(),'0')+int_to_string(i)+".png";
    //     Mat img = imread("/home/tejus/Documents/vision/lanes/image_2/"+image_name);
    //     detect_lanes(img);
    //     waitKey(1);
    // }

    // for(int i=0;i<=187;i++)
    // {
    //     cout<<i<<endl;
    //     image_name = string(10-int_to_string(i).length(),'0')+int_to_string(i)+".png";
    //     cout<<"/home/tejus/Documents/vision/lanes/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/"+image_name<<endl;
    //     Mat img = imread("/home/tejus/Documents/vision/lanes/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/"+image_name);
    //     cout<<img.rows<<","<<img.cols<<endl;
    //     detect_lanes(img);
    //     waitKey(1);
    // }

    // for(int i=0;i<=278;i++)
    // {
    //     cout<<i<<endl;
    //     image_name = string(10-int_to_string(i).length(),'0')+int_to_string(i)+".png";
    //     Mat img = imread("/home/tejus/Documents/vision/lanes/2011_09_30/2011_09_30_drive_0016_sync/image_02/data/"+image_name);
    //     cout<<img.rows<<","<<img.cols<<endl;
    //     detect_lanes(img);
    //     waitKey(1);
    // }

    // for(int i=0;i<=836;i++)
    // {
    //     cout<<i<<endl;
    //     image_name = string(10-int_to_string(i).length(),'0')+int_to_string(i)+".png";
    //     Mat img = imread("/home/tejus/Documents/vision/lanes/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/"+image_name);
    //     cout<<img.rows<<","<<img.cols<<endl;
    //     detect_lanes(img);
    //     waitKey(0);
    // }

    // for(int i=0;i<=389;i++)
    // {
    //     cout<<i<<endl;
    //     image_name = string(10-int_to_string(i).length(),'0')+int_to_string(i)+".png";
    //     Mat img = imread("/home/tejus/Documents/vision/lanes/2011_09_26/2011_09_26_drive_0032_sync/image_02/data/"+image_name);
    //     cout<<img.rows<<","<<img.cols<<endl;
    //     detect_lanes(img);
    //     waitKey(1);
    // }
    // eval_file.open("error.txt");
    // int total_frame=0;
    // Mat img = imread("/home/yash/AGV/lanedetector/kittiData/2011_09_26_drive_0001_sync/image_03/data/0000000022.png");
    // detect_lanes(img);

 //    eval_file.open("error.txt");
	

 //    for(int i=10;i<=389;i++)
 //    {
 //    	total_frame++;
 //        cout<<i<<endl;
 //        image_name = string(10-int_to_string(i).length(),'0')+int_to_string(i)+".png";
 //        Mat img = imread("/home/tejus/Documents/vision/lanes/2011_09_26/2011_09_26_drive_0032_sync/image_02/data/"+image_name);
 //        detect_lanes(img);

 //        try {evaluate(img,true,i);}
 //        catch(cv::Exception& e) {};
 //        waitKey(1);
 //    }
    int total_frame=0;
    for(int i=0;i<108;i++)
    {
        total_frame++;
        cout<<i<<endl;
        image_name = string(10-int_to_string(i).length(),'0')+int_to_string(i)+".png";
        cout<<image_name<<endl;
        Mat img = imread("/home/yash/AGV/lanedetector/kittiData/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/"+image_name);
        detect_lanes(img);
        waitKey(1);
    }


    waitKey(0);
    out.release();
    eval_file.close();
}