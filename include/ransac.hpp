#ifndef RANSAC
#define RANSAC


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>

/*
   parabolas are fit assuming top left as origin - x towards right and y downwards
 */

using namespace std;
using namespace cv;

Point centroid(float a,float c,Mat img);

float dist(Point A,Point B)
{
    return (sqrt(pow(A.x-B.x,2)+pow(A.y-B.y,2)));
}

//structure to define the Parabola parameters
typedef struct Parabola
{
    int numModel = 0;
    float a1 = 0.0;
    float c1 = 0.0;
    float a2 = 0.0;
    float c2 = 0.0;
} Parabola;

Parabola swap(Parabola param) {

    float temp1, temp2, temp3;
    temp1=param.a1;
    temp3=param.c1;

    param.a1=param.a2;
    param.c1=param.c2;

    param.a2=temp1;
    param.c2=temp3;

    return param;
}

//calculation of Parabola parameters based on 3 randonmly selected points
float get_a(Point p1, Point p2)
{
    int x1 = p1.x;
    int x2 = p2.x;
    int y1 = p1.y;
    int y2 = p2.y;

    float del = (y1 - y2)*(y1 + y2);
    float del_a = (x1 - x2);
    float a;
    a = del/(del_a);

    if(fabs(a)>500)
        return FLT_MAX;
    else
        return a;
}

float get_c(Point p1, Point p2)
{
    int x1 = p1.x;
    int y1 = p1.y;
    
    int x2 = p2.x;
    int y2 = p2.y;
        
    float del = (x1 - x2)*y2*y2;
    float del_a = (y1 - y2)*(y1 + y2);

    return (x2 - (del/(del_a)));
}

float min(float a, float b)
{
    if(a<=b)
        return a;
    return b;
}

//calculate distance of passed point from curve
float get_del(Point p, float a, float c)
{
    float predictedX = ((p.y*p.y)/(a) + c);
    float errorx = fabs(p.x - predictedX);

    //#TODO add fabs 
    float predictedY = sqrt(fabs(a*(p.x-c)));
    float errory = fabs(p.y - predictedY);

    return min(errorx, errory);
}

//removes both the lanes if they intersect within the image frame
bool isIntersectingLanes(Mat img, Parabola param) {
    float a1 = param.a1;
    float c1 = param.c1;

    float a2 = param.a2;
    float c2 = param.c2;

    if(a1==a2)
        return false;
    float x = (a1*c1 - a2*c2)/(a1-a2);
    
    //checks if intersection is within 

    float y_2 = a1*(x-c1);

    if (y_2 > 0 &&  sqrt(y_2) < (img.rows) && x > 0 && x < img.cols) return true;
    return false;

}

//choose Parabola parameters of best fit curve basis on randomly selected 3 points
Parabola ransac(vector<Point> ptArray, Parabola param, Mat img)
{
    int numDataPts = ptArray.size();
    
    Parabola bestTempParam;

    //initialising no. of lanes
    bestTempParam.numModel=2;
 

    int score_gl = 0;/* comm_count_gl = 0;*/
    // int metric_l_gl = 0, metric_r_gl = 0; 
    int score_l_gl = 0, score_r_gl = 0; 

    //check for no lane case here

    // loop of iterations
    for(int i = 0; i < iteration; i++)
    {
        int p1 = random()%ptArray.size(), p2 = random()%ptArray.size(), p3 = random()%ptArray.size(), p4 = random()%ptArray.size();
        

        if(p1==p2 || p1==p3 || p1==p4 || p3==p2 || p4==p2 || p3==p4){
            i--;
            continue;
        }
        //#TODO points with same x or y should not be passed in (p[0],p[1])&(p[2]&p[3]) 


        if(p2 == p1) p2 = random()%ptArray.size();
        
        if(p3 == p1 || p3 == p2) p3 = random()%ptArray.size();
        // TODO : p4 condition
        

        Point ran_points[4];
        ran_points[0] = ptArray[p1];
        ran_points[1] = ptArray[p2];
        ran_points[2] = ptArray[p3];
        ran_points[3] = ptArray[p4];

        int flag = 0;
        Point temp;

        for(int m = 0; m < 3; m++)
        {   
            for(int n = 0; n < 3 - m; n++)
            {
                if(ran_points[n].x > ran_points[n+1].x) 
                {
                    temp = ran_points[n];
                    ran_points[n] = ran_points[n+1];
                    ran_points[n+1] = temp;
                }
            }   
        }


        if(ran_points[0].x == ran_points[1].x || ran_points[2].x==ran_points[3].x || ran_points[0].y == ran_points[1].y || ran_points[2].y==ran_points[3].y){
            i--;
            continue;
        }

        Parabola tempParam; 
        tempParam.a1 = get_a(ran_points[0], ran_points[1]);
        tempParam.c1 = get_c(ran_points[0], ran_points[1]);


        tempParam.a2 = get_a(ran_points[2], ran_points[3]); 
        tempParam.c2 = get_c(ran_points[2], ran_points[3]);



        if (dist(centroid(tempParam.a1,tempParam.c1,img),centroid(tempParam.a2,tempParam.c2,img)) < 80.0)
            continue;

        if(fabs(tempParam.c1 - tempParam.c2) < 40.0)
            continue;

        // intersection only in top 3/8 part of the image taken
        if( isIntersectingLanes(img, tempParam)) {
            i--;
            continue;
        }

        //similar concavity of lanes

        
        // # rejected because many cases exist in which a better curve will get fit in opposite concavity
        // # and this will lead to bad lane fitting 
        // //similar concavity of lanes

        // if (tempParam.a1 * tempParam.a2 < 0) {
        //     continue;
        // }

        int score_common = 0;/*, comm_count = 0;*/
        int score_l_loc = 0, score_r_loc = 0;

        //looping over image
        for(int p = 0; p < ptArray.size(); p++)
        {

            int flag_l = 0; //for points on 1st curve
            int flag_r = 0; //for points on 2nd curve

            float dist_l = get_del(ptArray[p], tempParam.a1, tempParam.c1);

            if(dist_l < maxDist)
            {
                flag_l = 1;
            }

            float dist_r = get_del(ptArray[p], tempParam.a2, tempParam.c2);

            if(dist_r < maxDist)
            {
                flag_r = 1;
            }

            if(flag_l == 1 && flag_r == 1) {
                score_common++;
            }
            else {
                if (flag_l == 1) {
                    score_l_loc++;
                }
                if (flag_r == 1) {
                    score_r_loc++;
                }
            }
        } //end of loop over image

        // float lane_length_l = 1;
        // float lane_length_r = 1;
        // float metric_l = score_l_loc/lane_length_l;
        // float metric_r = score_r_loc/lane_length_r;

        if(score_r_loc==0 || score_l_loc==0)
            continue;

        if ((score_common/(score_common + score_l_loc + score_r_loc))*100 > common_inliers_thresh) {
            i--;
            continue;
        }

        // if(metric_l < metric_thresh && metric_r < metric_thresh){
        //     continue;
        // }

        if (score_l_loc + score_r_loc > score_gl) {
            // metric_l_gl=metric_l;
            // metric_r_gl=metric_r;

            score_l_gl=score_l_loc;
            score_r_gl=score_r_loc;
            score_gl = score_r_gl + score_l_gl;

            bestTempParam.a1=tempParam.a1;
            bestTempParam.c1=tempParam.c1;
            bestTempParam.a2=tempParam.a2;
            bestTempParam.c2=tempParam.c2;
        }


        /*
        if(score_loc > score_gl)
        {
            if(w < 100 && w > -100) continue;
            score_gl = score_loc;
                        score_l_gl = score_l_loc;
                        score_r_gl = score_r_loc;
            a_gl = a;
            lam_gl = lam;
            lam2_gl = lam2;
            w_gl = w;
            p1_g = p1;
            p2_g = p2;
            p3_g = p3;
            p4_g = p4;
            // comm_count_gl = comm_count;
            */


        } //end of iteration loop

    if(score_l_gl < minLaneInlier){
        bestTempParam.a1=0;
        bestTempParam.c1=0;
        bestTempParam.numModel--;
    }
    if(score_r_gl < minLaneInlier){
        bestTempParam.a2=0;
        bestTempParam.c2=0;
        bestTempParam.numModel--;
    }
    return bestTempParam;
}

Point centroid(float a,float c,Mat img)
{
    Point A;
    int i,j,x,y;
    int sum_x = 0,sum_y = 0,count=1;

    for(j=0;j<img.rows;j++)
    {
        y = img.rows-j;
        x = ((y*y)/(a) + c);

        if(x>=0 && x<img.cols)
        {
            sum_y+=y;
            sum_x+=x;
            count++;
        }
    }

    A.x=sum_x/count;
    A.y=sum_y/count;

    return A;
}



Parabola getRansacModel(Mat img,Parabola previous)
{
    //apply ransac for first time it will converge for one lane
    vector<Point> ptArray1;
    
    if (grid_white_thresh >= grid_size*grid_size) {
        grid_white_thresh = grid_size*grid_size -1;
    }

    Mat plot_grid(img.rows,img.cols,CV_8UC1,Scalar(0));
    for(int i=((grid_size-1)/2);i<img.rows-(grid_size-1)/2;i+=grid_size)
    {
        for(int j=((grid_size-1)/2);j<img.cols-(grid_size-1)/2;j+=grid_size)
        {
            int count=0;
            for(int x=(j-(grid_size-1)/2);x<=(j+(grid_size-1)/2);x++)
            {
                for(int y=(i-(grid_size-1)/2);y<=(i+(grid_size-1)/2);y++)
                {
                    if(img.at<uchar>(y,x)>128){
                        count++;
                        plot_grid.at<uchar>(i,j)=255;
                    }
                }
            }
            if(count>grid_white_thresh)
                ptArray1.push_back(Point(j , img.rows - i));
        }
    }

    namedWindow("grid",0);
    imshow("grid",plot_grid);

    //declare a Parabola vaiable to store the Parabola
    Parabola param;
    //get parameters of first Parabola form ransac function
    if(ptArray1.size() > minPointsForRANSAC)
    {
        param = ransac(ptArray1, param, img);
    }

    else {
        //param is already initialised to zero
    }


    //Lane classification based on previous frames


    //if two lanes
    if(param.numModel==2) {

        if(param.c2<param.c1)
        {
            param=swap(param);
        }
    }

    //if one lane, assign same as previous frame if it had one lane
    if(param.numModel==1)
    {
        if(previous.numModel==1)
        {
            //if prev frame had right lane
            if(previous.a1==0 && previous.c1==0)
            {
                //if current frame has left lane
                if(param.a2==0 && param.c2==0)
                {
                    param = swap(param);
                }
            }
            //if prev frame had left lane
            else if(previous.a2==0 && previous.c2==0)
            {
                //if current frame has right lane
                if(param.a1==0 && param.c1==0)
                {
                    param = swap(param);
                }
            }
        }

        if(previous.numModel==2)
        {
            Point A=centroid(previous.a1,previous.c1,img);
            Point B=centroid(previous.a2,previous.c2,img);
            Point C;

            //if current frame has right lane
            if(param.a1==0&&param.c1==0)
            {
                C=centroid(param.a2,param.c2,img);
                if(dist(A,C)<dist(B,C))
                {
                    param = swap(param);
                }
            }
            //if current frame has left lane
            else
            {
                C=centroid(param.a1,param.c1,img);
                if(dist(A,C)>dist(B,C))
                {
                    param = swap(param);
                }
            }
        }
    }


    return param;
}

Mat drawLanes(Mat img, Parabola lanes) {

    Mat fitLanes(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));

    vector<Point2f> left_lane, right_lane;
    float a1 = lanes.a1, a2 = lanes.a2, c1 = lanes.c1, c2 = lanes.c2;

    for (int j = 0; j < fitLanes.rows; j++){

        float x, y;
        if (a1 != 0 && c1 != 0) {
            y = fitLanes.rows - j;
            x = (y*y)/(a1) + c1;
            left_lane.push_back(Point2f(x, j));
        }

        if (a2 != 0 && c2 != 0) {
            y = fitLanes.rows - j;
            x = (y*y)/(a2) + c2;
            right_lane.push_back(Point2f(x, j));
        }

    }

    Mat left_curve(left_lane, true);
    left_curve.convertTo(left_curve, CV_32S); //adapt type for polylines
    polylines(fitLanes, left_curve, false, Scalar(255, 0, 0), 3, CV_AA);

    Mat right_curve(right_lane, true);
    right_curve.convertTo(right_curve, CV_32S); //adapt type for polylines
    polylines(fitLanes, right_curve, false, Scalar(0, 0, 255), 3, CV_AA);

    return fitLanes;
}


#endif
