#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <iostream>

using namespace std;
using namespace cv;

extern int lowThreshold;
extern int highThreshold;

string int_to_string(int n)
{
    std::ostringstream stm;
    stm << n;
    return stm.str();
}

int findIntersection(Vec4i l1, Vec4i l2)
{
	double m1, m2, c1, c2;

	m1=((double)l1[3]-l1[1])/((double)l1[2]-l1[0]);
	c1=(double)l1[3]-m1*l1[2];

	m2=((double)l2[3]-l2[1])/((double)l2[2]-l2[0]);
	c2=(double)l2[3]-m2*l2[2];

	double yi, xi;

	xi=(c1-c2)/(m2-m1);
	yi=m2*xi+c2;

	return (int)yi;
}

void transformPoints(Point2f* inputPoints, Point2f* transformedPoints, Mat transform, int npoints)
{
    for(int i=0; i<npoints; i++)
    {
        transformedPoints[i].x = (inputPoints[i].x*transform.at<double>(0, 0)+inputPoints[i].y*transform.at<double>(0, 1)+1.0*transform.at<double>(0, 2))/(inputPoints[i].x*transform.at<double>(2, 0)+inputPoints[i].y*transform.at<double>(2, 1)+1.0*transform.at<double>(2, 2));
        transformedPoints[i].y = (inputPoints[i].x*transform.at<double>(1, 0)+inputPoints[i].y*transform.at<double>(1, 1)+1.0*transform.at<double>(1, 2))/(inputPoints[i].x*transform.at<double>(2, 0)+inputPoints[i].y*transform.at<double>(2, 1)+1.0*transform.at<double>(2, 2));
    }
    return;
}

void transformLines(vector<Vec4i>& inputLines, vector<Vec4i>& transformedLines, Mat transform)
{
	for( size_t i = 0; i < inputLines.size(); i++ )
    {
        Vec4i l = inputLines[i];
        Point2f inputPoints[2], transformedPoints[2];
        inputPoints[0] = Point2f( l[0], l[1] );
        inputPoints[1] = Point2f( l[2], l[3] ); 
        transformPoints(inputPoints, transformedPoints, transform, 2);
        transformedLines.push_back(Vec4i(transformedPoints[0].x, transformedPoints[0].y, transformedPoints[1].x, transformedPoints[1].y));
    }
}

Mat getTemplateX(float sigma, int h, int w)
{
	float m=0;
	Mat templatex=Mat(h, w, CV_8UC1);
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			float x=j-w/2;
			templatex.at<uchar>(i, j)=275*(1/(sigma*sigma))*exp(-x*x/(2*sigma*sigma))*(1-x*x/(sigma*sigma))+128;
			m=max(m, 275*(1/(sigma*sigma))*exp(-x*x/(2*sigma*sigma))*(1-x*x/(sigma*sigma))+128);
		}
	//cout<<"max="<<m<<endl;
	return templatex;
}

Mat getTemplateX2(float sigma, int h, int w, float theta)
{
	//sigma = 2, 275; sigma = 1, 120; 1.6->270
	float m=0;
	Mat templatex=Mat(h, w, CV_8UC1);
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			float x=j-w/2 + tan(theta*3.14159/180)*(i-h/2);
			templatex.at<uchar>(i, j)=275*(1/(sigma*sigma))*exp(-x*x/(2*sigma*sigma))*(1-x*x/(sigma*sigma))+128;
			m=max(m, 275*(1/(sigma*sigma))*exp(-x*x/(2*sigma*sigma))*(1-x*x/(sigma*sigma))+128);
		}
	//cout<<theta<<endl;
	//cout<<"max: "<<m<<endl;
	// namedWindow("template", WINDOW_NORMAL);
	// imshow("template", templatex);
	return templatex;
}

void hysterisThreshold(Mat img, Mat& des, float lowThres, float highThres)
{
	Mat out=img-img;
	int i, j, k, l;

	int vis[img.rows][img.cols];
	for(i=0;i<img.rows;i++)
		for(j=0;j<img.cols;j++)
			vis[i][j]=-1;

	for(i=0;i<img.rows;i++)
		for(j=0;j<img.cols;j++)
		{
			if(img.at<float>(i,j)>=highThres && vis[i][j]!=1)
			{
				//init bfs to mark all nearby points
				queue<Point> q;
				q.push(Point(i,j));

				while(!q.empty())
				{
					Point current=q.front();
					q.pop();

					vis[i][j]=1;

					if(img.at<float>(current.x, current.y)>lowThres)
						out.at<float>(current.x, current.y)=img.at<float>(current.x, current.y);

					for(k=current.x-1;k<=current.x+1;k++)
						for(l=current.y-1;l<=current.y+1;l++)
						{
							if(k<0 || k>=img.rows || l<0 || l>=img.cols)
								continue;

							if(img.at<float>(k, l)>lowThres && vis[k][l]!=1)
							{
								q.push(Point(k, l));
								vis[k][l]=1;
							}
						}
				}
			}
		}
	des=out;
	return;
}