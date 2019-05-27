//input: gray-scale Mat and return a model(struct data type)
//main function: getMlesacModel

#ifndef MLESAC
#define MLESAC


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//structure to define the model parameters
typedef struct model
{
  int numModel = 0;
  float a1 = 0.0;
  float b1 = 0.0;
  float c1 = 0.0;
  float a2 = 0.0;
  float b2 = 0.0;
  float c2 = 0.0;
}model;


//set threshold for white color
#define wTh 50
//define no of iteration, max dist squre of pt from our estimated model
#define iteration 50
//iteration for EM algo for choosing best gamma
#define emIter 3
#define maxDist 5
//define sigma for gaussian distribution of inliers
#define sigmaVal 3.0
//define threshold distance to remove white pixel near lane1
#define removeDist 100
//define minimum number of points to be lie on a lane
#define minLaneInlier 750

//calculation of model parameters based on 3 randonmly selected points
float get_a(Point p1, Point p2, Point p3)
{
  int x1 = p1.x;
  int x2 = p2.x;
  int x3 = p3.x;
  int y1 = p1.y;
  int y2 = p2.y;
  int y3 = p3.y;
  float del = (y2 - y1)*(y3 - y2)*(y1 - y3);
  float del_a = (x2 - x1)*(y3 - y2) - (x3 - x2)*(y2 - y1);
  return(del_a / del);
}
float get_b(Point p1, Point p2, Point p3)
{
  int x1 = p1.x;
  int x2 = p2.x;
  int x3 = p3.x;
  int y1 = p1.y;
  int y2 = p2.y;
  int y3 = p3.y;
  float del = (y2 - y1)*(y3 - y2)*(y1 - y3);
  float del_b = (x3 - x2)*((y2*y2) - (y1*y1)) - (x2 - x1)*((y3*y3) - (y2*y2));
  return(del_b / del);
}
float get_c(Point p, float a, float b)
{
  int x = p.x;
  int y = p.y;
  return(x - (a*y*y) - (b*y));
}

//calculation of error b/w actual and estimated y
float get_delX(Point p, float a, float b, float c)
{
  float predictedX = (a*(p.y*p.y) + b*p.y + c);
  return abs(p.x - predictedX);
}

//calculation of inlier prob based on gaussian distribution
float inlierProb(float error, float sigma, float gamma)
{
  float inlierP = gamma*(exp(-(error*error)/(2*sigma*sigma))/(sigma*2.50662827463));
  return inlierP;
}

//calculation of outlier prob based on unform distribution
float outlierProb(float gamma, float globMaxResid)
{
  float outlierP = (1-gamma)/globMaxResid;
  return outlierP;
}

//choose model parameters of best fit curve basis on randomly selected 3 points
model mlesac(vector<Point> ptArray, float globMaxResid, model param)
{
  int dataPt = ptArray.size();
  model tempParam;
  float MinPenalty;
  int maxInlier = 0;
  float sigma = sigmaVal;
  for(int i = 0; i < iteration; ++i)
  {
    float gamma = 0.5; //initialisation
    float currPenalty = 0;
    int d1 = rand() % dataPt;
    int d2 = rand() % dataPt;
    int d3 = rand() % dataPt;
    Point p1 = ptArray[d1];
    Point p2 = ptArray[d2];
    Point p3 = ptArray[d3];
    if((p1.x == p2.x) || (p2.x == p3.x) || (p3.x == p1.x)||(p1.y == p2.y) || (p2.y == p3.y) || (p3.y == p1.y))
    {
      continue;
    }
    //get the model parameters
    float temp_a = get_a(p1, p2, p3);
    float temp_b = get_b(p1, p2, p3);
    float temp_c = get_c(p1, temp_a, temp_b);
    //count inliers on predicted model
    int tempInlier = 0;

    //choose best gamma by EM algorithm
    for(int k = 0; k < emIter; ++k)
    {
      float inlierPB = 0;
      for(int j = 0; j < dataPt; ++j)
      {
        Point z = ptArray[j];
        float error = get_delX(z, temp_a, temp_b, temp_c);
        if(error < maxDist)
        {
          ++tempInlier;
        }
        float inlierP = inlierProb(error, sigma, gamma);
        float outlierP = outlierProb(gamma, globMaxResid);
        inlierPB += (inlierP)/(inlierP + outlierP);
        if(k == emIter - 1)
        {
          currPenalty -= log(inlierP + outlierP);
        }
      }
      gamma = inlierPB/dataPt;
    }
    if((MinPenalty > currPenalty) || (i == 0))
    {
      maxInlier = tempInlier;
      MinPenalty = currPenalty;
      tempParam.a1 = temp_a;
      tempParam.b1 = temp_b;
      tempParam.c1 = temp_c;
    }
  }
  if(maxInlier > minLaneInlier)
  {
    if(param.numModel == 0)
    {
      param.a1 = tempParam.a1;
      param.b1 = tempParam.b1;
      param.c1 = tempParam.c1;
    }
    else if(param.numModel == 1)
    {
      param.a2 = tempParam.a1;
      param.b2 = tempParam.b1;
      param.c2 = tempParam.c1;
    }
    param.numModel += 1;
  }
  return param;
}

//Check wheather a point is near lane1
bool IsNearLane1(model param1, Point p)
{
  float dist = get_delX(p, param1.a1, param1.b1, param1.c1);
  if(dist < removeDist)
  {
    return true;
  }
  else
  {
    return false;
  }
}


model getMlesacModel(Mat img)
{
  float diaLen = sqrt(pow(img.rows, 2) + pow(img.cols, 2));
  //apply mlesac for first time it will converge for one lane
  vector<Point> ptArray1;
  for(int i = 0; i < img.rows; ++i)
  {
    for(int j = 0; j < img.cols; ++j)
    {
      int wVal = img.at<uchar>(i,j);
      if(wVal > wTh)
      {
        Point pt;
        pt.x = j;
        pt.y = i;
        ptArray1.push_back(pt);
      }
    }
  }

  //declare a model vaiable to store the model
  model param;

  //get parameters of first model form ransac function
  if(ptArray1.size() > 500)
  {
    param = mlesac(ptArray1, diaLen, param);
  }

  //Remove white pixel form image near lane1 and apply ransac to get lane2
  if(param.numModel > 0)
  {
    vector<Point> ptArray2;
    for(int i = 0; i < img.rows; ++i)
    {
      for(int j = 0; j < img.cols; ++j)
      {
        Point q;
        q.x = j;
        q.y = i;
        int wVal = img.at<uchar>(i,j);
        if(wVal > wTh)
        {
          if(!IsNearLane1(param, q))
          {
            ptArray2.push_back(q);
          }
        }
      }
    }

    //get parameters of second model form mlesac function
    if(ptArray2.size() > 500)
    {
      param = mlesac(ptArray2, diaLen, param);
    }
  }

  return param;
}


#endif
