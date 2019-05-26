#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

Mat top_view(Mat img, Mat transform) {

	Mat topview;
    warpPerspective(img, topview, transform, Size(800,1000));

    return topview;
}

Mat front_view(Mat img, Mat transform) {
	Mat frontview;
	warpPerspective(img, frontview, transform.inv(), Size(1920, 1200), INTER_NEAREST, BORDER_CONSTANT);

	return frontview;
}

