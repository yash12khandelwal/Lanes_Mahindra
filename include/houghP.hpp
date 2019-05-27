#ifndef _HOUGHP_HPP_
#define _HOUGHP_HPP_

#include "opencv/cv.h"
#include <opencv2/highgui/highgui.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

void HoughLinesP2( Mat& image, vector<Vec4i>& lines, vector<int>& len,
                         float rho, float theta, int threshold,
                         int lineLength, int lineGap);

#endif