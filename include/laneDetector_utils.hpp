#ifndef _LANEDETECTOR_UTILS_HPP_
#define _LANEDETECTOR_UTILS_HPP_

#include "opencv/cv.h"
#include <opencv2/highgui/highgui.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

extern int lowThreshold;
extern int highThreshold;

string int_to_string(int n);
int findIntersection(Vec4i l1, Vec4i l2);
void transformPoints(Point2f* inputPoints, Point2f* transformedPoints, Mat transform2, int npoints);
void transformLines(vector<Vec4i>& inputLines, vector<Vec4i>& transformedLines, Mat transform);
Mat getTemplateX(float sigma, int h, int w);
Mat getTemplateX2(float sigma, int h, int w, float theta);
void hysterisThreshold(Mat img, Mat& des, float lowThres, float highThres);
#endif


/*
Sign Conventions:

findIntersections: gives the distance of the intersection of the two lines from the top of the segment

cv::Line takes points as
--------->	first co-ordinate
|
|
|
|
|
second co-ordinate

*/