#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>


using namespace std;
using namespace cv;

Mat transform = (Mat_<double>(3, 3) << -0.2845660084796459, -0.6990548252793777, 691.2703423570697, -0.03794262877137361, -2.020741261264247, 1473.107653024983, -3.138403683957707e-05, -0.001727021397398348, 1);

Mat top_view(Mat img) {

	Mat topview;
    warpPerspective(img, top_view, transform, Size(800,1000));

    return top_view;
}

Mat front_view(Mat img) {
	Mat front_view;
	warpPerspective(img, front_view, transform.inv(), Size(1920, 1200), INTER_NEAREST, BORDER_CONSTANT)

	return front_view;
}

