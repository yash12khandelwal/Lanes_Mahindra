#ifndef RANSAC
#define RANSAC


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>

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
#define iteration 100
#define maxDist 5
//define threshold distance to remove white pixel near lane1
#define removeDist 100
//define minimum number of points to be lie on a lane
#define minLaneInlier 500

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

//choose model parameters of best fit curve basis on randomly selected 3 points
model ransac(vector<Point> ptArray, model param)
{
  int dataPt = ptArray.size();
  int maxInlier = 0;
  model tempParam;
  for(int i = 0; i < iteration; ++i)
  {
    int d1 = rand() % dataPt;
    int d2 = rand() % dataPt;
    int d3 = rand() % dataPt;
    Point p1 = ptArray[d1];
    Point p2 = ptArray[d2];
    Point p3 = ptArray[d3];
    if(d1==d2 || d2==d3 || d1==d3)
    {
      continue;
    }
    //get the model parameters
    float temp_a = get_a(p1, p2, p3);
    float temp_b = get_b(p1, p2, p3);
    float temp_c = get_c(p1, temp_a, temp_b);

    //count inliers on predicted model
    int tempInlier = 0;
    for(int j = 0; j < dataPt; ++j)
    {
      Point z = ptArray[j];
      float error = get_delX(z, temp_a, temp_b, temp_c);
      if(error < maxDist)
      {
        ++tempInlier;
      }
    }
    if((tempInlier > maxInlier) || (i == 0))
    {
      maxInlier = tempInlier;
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

float distance_nikaal(Point A,Point B)
{
	return (sqrt(pow(A.x-B.x,2)+pow(A.y-B.y,2)));
}

Point centroid(float a,float b,float c,Mat img)
{
	Point A;
	int i,j,x,y;
	int sum_x,sum_y,count=0;

	for(j=0;j<img.rows;j++)
	{
		x=a*j*j+b*j+c;

		if(x>=0 && x<img.cols)
		{
			sum_y+=j;
			sum_x+=x;
			count++;
		}
	}

	A.x=sum_x/count;
	A.y=sum_y/count;

	return A;

}

model getRansacModel(Mat img,model previous)
{
  //apply ransac for first time it will converge for one lane
  vector<Point> ptArray1;
  Point A,B,C;

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

  cout << "Total number of points: " << ptArray1.size() << endl;
  //declare a model vaiable to store the model
  model param;

  //get parameters of first model form ransac function
  if(ptArray1.size() > 500) //#TODO why 500?
  {
    param = ransac(ptArray1, param);
  }

  //Remove white pixel form image near lane1 and apply ransac to get lane2
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
  cout << "Total number of points after removal: " << ptArray2.size() << endl;

  float temp1,temp2,temp3;
  //get parameters of second model form ransac function
  if(ptArray2.size() > 500)
  {
    param = ransac(ptArray2, param);
  }
  if(param.numModel==2)	
     if(param.a1*img.rows/2*img.rows/2+param.b1*img.rows/2+param.c1>param.a2*img.rows/2*img.rows/2+param.b2*img.rows/2+param.c2)
     {
      	temp1=param.a1;
      	temp2=param.b1;
      	temp3=param.c1;
      	param.a1=param.a2;
      	param.b1=param.b2;
      	param.c1=param.c2;
      	param.a2=temp1;
      	param.b2=temp2;
      	param.c2=temp3;
     }

//if current lane is single
  if(param.numModel==1)
  {
    //previous lane is single
  	if(previous.numModel==1)
  	{



      //previous lane is right
  		if(previous.a1==0&&previous.b1==0&&previous.c1==0)
  		{
        //if current lane is left
  			if(param.a2==0&&param.b2==0&&param.c2==0)
  			{
  				param.a2 = param.a1;
  				param.b2 = param.b1;
  				param.c2 = param.c1;
          
          param.a1 = 0;
          param.b1 = 0;
          param.c1 = 0;
			  }
  		}

      //if previous was left
  		else if(previous.a2==0&&previous.b2==0&&previous.c2==0)
  		{
          //if current is right
  			if(param.a1==0&&param.b1==0&&param.c1==0)
  			{
  	  		param.a1 = param.a2;
          param.b1 = param.b2;
          param.c1 = param.c2;
          
          param.a2 = 0;
          param.b2 = 0;
          param.c2 = 0;
			  }
  		}


  	}




    //if 2 previous lanes
  	if(previous.numModel==2)
  	{
  		A=centroid(previous.a1,previous.b1,previous.c1,img);
  		B=centroid(previous.a2,previous.b2,previous.c2,img);

      //if current lane is right
  		if(param.a1==0&&param.b1==0&&param.c1==0)
  		{
  			C=centroid(param.a2,param.b2,param.c2,img);
	  		if(distance_nikaal(A,C)<distance_nikaal(B,C))
	  		{

  				param.a1=param.a2;
  				param.b1=param.b2;
  				param.c1=param.c2;

  				param.a2=0;
  				param.b2=0;
  				param.c2=0;
	  		}
  		}
  		else
  		{
  			C=centroid(param.a1,param.b1,param.c1,img);
	  		if(distance_nikaal(A,C)>distance_nikaal(B,C))
	  		{
  				param.a2=param.a1;
          param.b2=param.b1;
          param.c2=param.c1;

          param.a1=0;
  				param.b1=0;
  				param.c1=0;
        
	  		}
  		}
  	}
  }
  cout << "Total number of lanes detected: " << param.numModel << endl;

  if (param.numModel == 1) {
    if (param.a1 == 0 && param.b1 == 0 && param.c1 == 0) {
      cout << "Right lane detected" << endl;
    }
    else {
      cout << "Left lane detected" << endl;
    }
  }
  return param;
}

#endif
