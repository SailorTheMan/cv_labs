#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

int th = 0;
Mat img, thimg;
void proc_img(int,void* user_data){
	int* th_type = (int*)user_data;
	threshold(img, thimg, th, 255, *th_type);
	imshow("thimg", thimg);
	vector<vector<Point>> cnts;
	findContours(thimg, cnts, RETR_LIST, CHAIN_APPROX_NONE);
	Mat draw = Mat::zeros(thimg.size(), CV_8UC3);
	int maxAreaInd = 0;
	double maxArea = 0;
	for (size_t i = 0; i < cnts.size(); i++)
	{
		double area = contourArea(cnts[i]);
		if (area > maxArea){
			maxArea = area;
			maxAreaInd = i;
		}
	}
	Moments mnts = moments(cnts[maxAreaInd]);
	int centerX = mnts.m10 / mnts.m00;
	int centerY = mnts.m01 / mnts.m00;
	circle(draw, Point(centerX, centerY), 5, Scalar(255, 0, 0));
	drawContours(draw, cnts, maxAreaInd, Scalar(0, 0, 255));
	imshow("cntimg", draw);
}

int main(int argc, char** argv){
	string fn;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/roboti/roi_robotov.jpg";
	img = imread(fn,0);
	imshow(fn, img);

	int th_type = THRESH_BINARY;

	proc_img(0,&th_type);
	createTrackbar("th", "thimg", &th, 255, proc_img, &th_type);
	waitKey();
}