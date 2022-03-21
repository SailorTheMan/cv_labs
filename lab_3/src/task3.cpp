#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;

int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value, minArea=0;


void find_robots(Mat &src, vector<vector<Point>> &contours, Scalar &range_min, Scalar &range_max, int minArea=0){
    Mat thimg;
    inRange(src, range_min, range_max, thimg);
    Mat opened, closed, dilated;
    morphologyEx(thimg, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    morphologyEx(opened, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(10, 10)));
    morphologyEx(closed, dilated, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(5, 5)));
    vector<vector<Point>> cnts;
    findContours(dilated, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i]) < minArea)
            continue;
        contours.push_back(cnts[i]);
    }
}

int main(int argc, char** argv){
	namedWindow("thimg");
	string fn;
	Mat img, thimg;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/roboti/roi_robotov.jpg";
	img = imread(fn);
	imshow(fn, img);
    Mat bgrImg, hsvImg, gray;

    Scalar red_low_0(15, 242, 83); 
    Scalar red_high_0(38, 255, 138);
    Scalar red_low_1(151, 184, 224);
    Scalar red_high_1(180, 224, 255);

    Scalar green_low(90, 133, 0);
    Scalar green_high(119, 255, 255);

    Scalar blue_low(73, 139, 0);
    Scalar blue_high(89, 218, 255);

    Scalar blue_clr(255, 0, 0);
    Scalar green_clr(0, 255, 0);
    Scalar red_clr(0, 0, 255);
    cvtColor(img, bgrImg, COLOR_YUV2BGR);
    cvtColor(bgrImg, hsvImg, COLOR_BGR2HSV);
    
    vector<vector<Point>> contours_green;
    find_robots(hsvImg, contours_green, green_low, green_high);
    vector<vector<Point>> contours_blue;
    find_robots(hsvImg, contours_blue, blue_low, blue_high);
    vector<vector<Point>> contours_red;
    find_robots(hsvImg, contours_red, red_low_0, red_high_0);
    find_robots(hsvImg, contours_red, red_low_1, red_high_1);

    Mat draw = img.clone();
    vector<Point> greenRobotsCenters;
    for (size_t i = 0; i < contours_green.size(); i++)
    {
        Moments mnts = moments(contours_green[i]);
        int centerX = mnts.m10 / mnts.m00;
        int centerY = mnts.m01 / mnts.m00;
        greenRobotsCenters.push_back(Point(centerX, centerY));
        drawContours(draw, contours_green, i, green_clr, 3);
    }
    vector<Point> blueRobotsCenters;
    for (size_t i = 0; i < contours_blue.size(); i++)
    {
        Moments mnts = moments(contours_blue[i]);
        int centerX = mnts.m10 / mnts.m00;
        int centerY = mnts.m01 / mnts.m00;
        blueRobotsCenters.push_back(Point(centerX, centerY));
        drawContours(draw, contours_blue, i, blue_clr, 3);
    }
    vector<Point> redRobotsCenters;
    for (size_t i = 0; i < contours_red.size(); i++)
    {
        Moments mnts = moments(contours_red[i]);
        int centerX = mnts.m10 / mnts.m00;
        int centerY = mnts.m01 / mnts.m00;
        redRobotsCenters.push_back(Point(centerX, centerY));
        drawContours(draw, contours_red, i, red_clr, 3);
    }
    imshow("draw", draw);

    // TASK3.2_______________________________________________________________________
    gray = imread(fn, 0);
    
    threshold(gray, thimg, 247, 255, THRESH_BINARY);
    imshow("thimg", thimg);

    vector<vector<Point>> cnts;
    findContours(thimg, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    double maxArea = 0;
    int lampCenterX = 0;
    int lampCenterY = 0;
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i]) > maxArea){
            Moments mnts = moments(cnts[i]);
            lampCenterX = mnts.m10 / mnts.m00;
            lampCenterY = mnts.m01 / mnts.m00;
        }
    }
    circle(draw, Point(lampCenterX, lampCenterY), 3, Scalar(0, 0, 0), 3);
    float minDistance = 0;

    Point closestGreen;
    for (size_t i = 0; i < greenRobotsCenters.size(); i++)
    {
        float distance = sqrt((greenRobotsCenters[i].x - lampCenterX) * (greenRobotsCenters[i].x - lampCenterX) + 
                            (greenRobotsCenters[i].y - lampCenterY) * (greenRobotsCenters[i].y - lampCenterY));
        if ((i == 0) || (distance < minDistance)){
            minDistance = distance;
            closestGreen.x = greenRobotsCenters[i].x;
            closestGreen.y = greenRobotsCenters[i].y;
        }
    }
    circle(draw, closestGreen, 3, Scalar(255, 255, 255), 3);

    minDistance = 0;
    
    Point closestBlue;
    for (size_t i = 0; i < blueRobotsCenters.size(); i++)
    {
        float distance = sqrt((blueRobotsCenters[i].x - lampCenterX) * (blueRobotsCenters[i].x - lampCenterX) + 
                            (blueRobotsCenters[i].y - lampCenterY) * (blueRobotsCenters[i].y - lampCenterY));
        if ((i == 0) || (distance < minDistance)){
            minDistance = distance;
            closestBlue.x = blueRobotsCenters[i].x;
            closestBlue.y = blueRobotsCenters[i].y;
        }
    }
    circle(draw, closestBlue, 3, Scalar(255, 255, 255), 3);

    minDistance = 0;
    Point closestRed;
    for (size_t i = 0; i < redRobotsCenters.size(); i++)
    {
        float distance = sqrt((redRobotsCenters[i].x - lampCenterX) * (redRobotsCenters[i].x - lampCenterX) + 
                            (redRobotsCenters[i].y - lampCenterY) * (redRobotsCenters[i].y - lampCenterY));
        if ((i == 0) || (distance < minDistance)){
            minDistance = distance;
            closestRed.x = redRobotsCenters[i].x;
            closestRed.y = redRobotsCenters[i].y;
        }
    }
    circle(draw, closestRed, 3, Scalar(255, 255, 255), 3);
    
    imshow("closest", draw);

    waitKey();
}

//red H:15-38 S:242-255 V:83-138 && H:151-180 S:184-224 V:243-255
//green H:90-119 S:133-255 V:0-255
//blue H:73-89 S:139-218 V:175-255