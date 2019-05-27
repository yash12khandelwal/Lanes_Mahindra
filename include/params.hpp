#ifndef PARAMS
#define PARAMS

bool is_debug = false;

int iteration = 100;  //define no of iteration, max dist squre of pt from our estimated Parabola2
int maxDist = 300; //define threshold distance to remove white pixel near lane1
int minLaneInlier = 1500; // 2000 for night
int common_inliers_thresh = 10;
int minPointsForRANSAC = 500;
int grid_size = 3;
int grid_white_thresh = 3;

float pixelPerMeter = 134;

int horizon = 500;
int horizon_offset = 200;

int transformedPoints0_lowerbound = 30;
int transformedPoints0_upperbound = 800;

int point1_y = 30;
int point2_x = 100;

int h = 30;
int w = 10;
float variance = 2.1;

float yshift = 0.60; // distance from first view point to lidar in metres

float hysterisThreshold_min = 0.39;
float hysterisThreshold_max = 0.45;

int y = 400;
int lane_width = 320;
int k1 = 50;
int k2 = 50;

int medianBlurkernel = 3; //kernel size of medianBlur for cleaning intersectionImages
int neighbourhoodSize = 25; //neighbourhood size or block size for adaptive thresholding
int constantSubtracted = -30; //constant subtracted during adaptive thresholding

#endif

