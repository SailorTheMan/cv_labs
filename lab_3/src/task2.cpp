#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;

int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", "thimg", low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", "thimg", high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", "thimg", low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", "thimg", high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", "thimg", low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", "thimg", high_V);
}

int main(int argc, char** argv){
	namedWindow("thimg");
	string fn;
	Mat img, thimg;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/teplovizor/ntcs_quest_measurement.png";
	img = imread(fn);
	imshow(fn, img);

	int th_type = THRESH_BINARY;

	createTrackbar("Low H", "thimg", &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", "thimg", &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", "thimg", &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", "thimg", &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", "thimg", &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", "thimg", &high_V, max_value, on_high_V_thresh_trackbar);
	while(true){
		Mat bgrImg, hsvImg;
		cvtColor(img, bgrImg, COLOR_YUV2BGR);
		cvtColor(bgrImg, hsvImg, COLOR_BGR2HSV);
		inRange(hsvImg, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), thimg);
		imshow("thimg", thimg);
		vector<vector<Point>> cnts;
		findContours(thimg, cnts, RETR_LIST, CHAIN_APPROX_NONE);
		Mat draw = Mat::zeros(thimg.size(), CV_8UC3);
		for (size_t i = 0; i < cnts.size(); i++)
		{
			Moments mnts = moments(cnts[i]);
			int centerX = mnts.m10 / mnts.m00;
			int centerY = mnts.m01 / mnts.m00;
			circle(draw, Point(centerX, centerY), 2, Scalar(255, 0, 0), 2);
			drawContours(draw, cnts, i, Scalar(0, 0, 255));
		}
		imshow("cntimg", draw);
		waitKey(5);
	}
	
}