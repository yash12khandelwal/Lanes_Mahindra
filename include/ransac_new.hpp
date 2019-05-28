#ifndef RANSAC_NEW
#define RANSAC_NEW


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
    // float b2 = 0.0;
    float c2 = 0.0;
} Parabola;

Parabola swap(Parabola param) {

    float temp1, temp2, temp3;
    temp1=param.a1;
    // temp2=param.b1;
    temp3=param.c1;

    param.a1=param.a2;
    // param.b1=param.b2;
    param.c1=param.c2;

    param.a2=temp1;
    // param.b2=temp2;
    param.c2=temp3;

    return param;
}

Parabola classify_lanes(Mat img,Parabola present,Parabola previous)
{
    float a1=present.a1;
    float a2=present.a2;
    float c1=present.c1;
    float c2=present.c2;
    int number_of_lanes=present.numModel;

    if(number_of_lanes==2)
    {
        if(c2<c1)
        {
            present=swap(present);
            return present;
        }
        else 
            return present;
    }

    else if(number_of_lanes==1)
    {
        //if intersection on left or right lane possible
        if(a1*c1<0 && a1*(img.cols-c1)>0)
        {
            float y1=sqrt(-1.0*a1*c1);
            float y2=sqrt(a1*(img.cols-c1));

            if(y1>0 && y1<img.rows && y2>0 && y2<img.rows)
            {
                return previous;
            }

        }

        if(a2*c2<0 && a2*(img.cols-c2)>0)
        {
            float y1=sqrt(-1.0*a2*c2);
            float y2=sqrt(a2*(img.cols-c2));

            if(y1>0 && y1<img.rows && y2>0 && y2<img.rows)
            {
                return previous;
            }
        }

        if((c1!=0 && c1>(2*img.cols/5) && c1<(3*img.cols/5))|| (c2!=0 && c2>(2*img.cols/5) && c2<(3*img.cols/5)))
        {
            return previous;
        }

        if(c1!=0 && c1>(3*img.cols/5))
        {
            present=swap(present);
            return present;
        }

        else if(c2!=0 && c2<(2*img.cols/5))
        {
            present=swap(present);
            return present;
        }


    }

}

//calculation of Parabola parameters based on 3 randonmly selected points
float get_a(Point p1, Point p2)
{
    int x1 = p1.x;
    int x2 = p2.x;
    // int x3 = p3.x;
    int y1 = p1.y;
    int y2 = p2.y;
    // int y3 = p3.y;

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
    float x = (a1*c1 - a2*c2)/(a1-a2 );
   
    //checks if intersection is within

    float y_2 = a1*(x-c1);
    cout<<"y_2 : "<<y_2<<" x : "<<x<<endl;

    if (y_2 >= 0 &&  sqrt(y_2) <= (img.rows) && x > 0 && x < img.cols) {
        cout<<"intersectttttttttttttttttt hooooooooooooooo gyaaaaaaaaaaaaa"<<endl;
        return true;
    }

    return false;
}



//choose Parabola parameters of best fit curve basis on randomly selected 3 points



Parabola ransac(vector<Point> ptArray, Parabola bestTempParam, Mat img)
{
    int numDataPts = ptArray.size();

    int score_gl = 0;

    // loop of iterations
    for(int i = 0; i < iteration; i++)
    {
        //cout<<"i : "<<i<<endl;
        int score_loc=0;
        int p1 = random()%ptArray.size(), p2 = random()%ptArray.size();
       

        if(p1==p2 ){
            // i--;
            continue;
        }

        Point ran_points[2];
        ran_points[0] = ptArray[p1];
        ran_points[1] = ptArray[p2];


        int flag = 0;
        Point temp;

        if(ran_points[0].x == ran_points[1].x || ran_points[0].y == ran_points[1].y){
            // i--;
            continue;
        }

        //cout<<"mario"<<endl;

        Parabola tempParam;
        tempParam.a1 = get_a(ran_points[0], ran_points[1]);
        tempParam.c1 = get_c(ran_points[0], ran_points[1]);

        //a for second curve be same as first
        if(bestTempParam.numModel==1)
        {
            tempParam.a1=bestTempParam.a1;
        }

        //cout<<"doraemon"<<endl;

        //cout<<"a1: "<<tempParam.a1<<" c1: "<<tempParam.c1<<endl;

        // cout << "Centroid Dif : " << dist(centroid(tempParam.a1,tempParam.c1,img),centroid(tempParam.a2,tempParam.c2,img)) << endl;

        //looping over image
        for(int p = 0; p < ptArray.size(); p++)
        {

            float dist_l = get_del(ptArray[p], tempParam.a1, tempParam.c1);

            if(dist_l < maxDist)
            {
                 score_loc++;
            }

        } //end of loop over image


        //cout<<"numModel : "<<bestTempParam.numModel<<" score_loc : "<<score_loc<<endl;

        if (score_loc > score_gl) {

            score_gl = score_loc;
            //cout<<"score_gl XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx : "<<score_gl<<endl;
            if(bestTempParam.numModel==0)
            {
                bestTempParam.a1=tempParam.a1;
                bestTempParam.c1=tempParam.c1;
            }
            else
            {
                bestTempParam.a2=tempParam.a1;
                bestTempParam.c2=tempParam.c1;
            }
        }

    } //end of iteration loop

    bestTempParam.numModel++;

    if(score_gl < minLaneInlier){
        if(bestTempParam.numModel==1)
        {
            bestTempParam.a1=0;
            bestTempParam.c1=0;
        }
        else if(bestTempParam.numModel==2)
        {
            bestTempParam.a2=0;
            bestTempParam.c2=0;
        }
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

double wTh = 128;

Parabola getRansacModel(Mat img,Parabola previous)
{
    //apply ransac for first time it will converge for one lane
    vector<Point> ptArray1,ptArray2;
   
    if (grid_white_thresh >= grid_size*grid_size) {
        grid_white_thresh = grid_size*grid_size -1;
    }

    Mat plot_grid(img.rows,img.cols,CV_8UC1,Scalar(0));
    // cout << "grid_size: " << grid_size << endl;
    // cout << "grid_white_thresh: " << grid_white_thresh << endl;
    for(int i=((grid_size-1)/2);i<img.rows-(grid_size-1)/2;i+=grid_size)
    {
        for(int j=((grid_size-1)/2);j<img.cols-(grid_size-1)/2;j+=grid_size)
        {
            int count=0;
            for(int x=(j-(grid_size-1)/2);x<=(j+(grid_size-1)/2);x++)
            {
                for(int y=(i-(grid_size-1)/2);y<=(i+(grid_size-1)/2);y++)
                {
                    if(img.at<uchar>(y,x)>wTh){
                        count++;
                        plot_grid.at<uchar>(i,j)=255;
                    }
                }
            }
            if(count>grid_white_thresh)
                ptArray1.push_back(Point(j , img.rows - i));
        }
    }
    //cout << "**********************ptArray1: ***************" << ptArray1.size() << endl;

    namedWindow("grid",0);
    imshow("grid",plot_grid);

    //declare a Parabola vaiable to store the Parabola
    Parabola param;

    //cout<<"ransac th :"<<minPointsForRANSAC<<endl;
    //get parameters of first Parabola form ransac function
    if(ptArray1.size() > minPointsForRANSAC)
    {
        //cout<<"No of pts : "<<ptArray1.size()<<endl;
        param = ransac(ptArray1, param, img);
    }

    if(param.numModel==1)
    {
        for(int i=0;i<ptArray1.size();i++)
        {
            if(get_del(ptArray1[i],param.a1,param.c1)>maxDist)
            {
                ptArray2.push_back(ptArray1[i]);
            }
        }
        if(ptArray2.size()>minLaneInlier)
        {
            //cout<<"ptArray2 size : "<<ptArray2.size()<<endl;
            param=ransac(ptArray2,param,img);
        }
    }

    //cout<<"Finally Number of lines detected : "<<param.numModel<<endl;
    //Lane classification based on previous frames

    //common_inliers is useless here

    //if two lanes

    if(param.numModel==2)
    {
        if(fabs(param.c2-param.c1)<60)
        {
            param.a2=0;
            param.c2=0;
            param.numModel--;
            return param;
        }

        if(isIntersectingLanes(img,param))
        {
            return previous;
        }

    }

    param=classify_lanes(img,param,previous);

   

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

Mat drawLanes_white(Mat img, Parabola lanes) {

    vector<Point2f> left_lane, right_lane;
    float a1 = lanes.a1, a2 = lanes.a2, c1 = lanes.c1, c2 = lanes.c2;

    for (int j = 0; j < img.rows; j++){

        float x, y;
        if (a1 != 0 && c1 != 0) {
            y = img.rows - j;
            x = (y*y)/(a1) + c1;
            left_lane.push_back(Point2f(x, j));
        }

        if (a2 != 0 && c2 != 0) {
            y = img.rows - j;
            x = (y*y)/(a2) + c2;
            right_lane.push_back(Point2f(x, j));
        }

    }

    Mat left_curve(left_lane, true);
    left_curve.convertTo(left_curve, CV_32S); //adapt type for polylines
    polylines(img, left_curve, false, Scalar(255, 0, 0), 3, CV_AA);

    Mat right_curve(right_lane, true);
    right_curve.convertTo(right_curve, CV_32S); //adapt type for polylines
    polylines(img, right_curve, false, Scalar(0, 0, 255), 3, CV_AA);

    return img;
}

#endif