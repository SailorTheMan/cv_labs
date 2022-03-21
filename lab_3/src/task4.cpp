#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

int main(int argc, char** argv){
	string fn;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/gk/gk_tmplt.jpg";
    Mat templ = imread(fn, IMREAD_GRAYSCALE);
    Mat templ_th;
    threshold(templ, templ_th, 200, 255, THRESH_BINARY);
    vector<vector<Point>> cnts_template;
    findContours(templ_th, cnts_template, RETR_LIST, CHAIN_APPROX_NONE);
    Mat draw = Mat::zeros(templ.size(), CV_8UC3);
    drawContours(draw, cnts_template, 0, Scalar(255, 0, 0));
    imshow("draw", draw);

    Mat img = imread("../img_zadan/gk/gk.jpg", IMREAD_GRAYSCALE);
    Mat gray, opened, closed, dilated;
    threshold(img, gray, 230, 255, THRESH_BINARY);
    imshow("img", gray);
    morphologyEx(gray, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    morphologyEx(opened, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    imshow("morph", closed);
    vector<vector<Point>> cnts;
    Mat wrench_draw = imread("../img_zadan/gk/gk.jpg");
    findContours(closed, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i])>1000000)
            continue;
        double diff = matchShapes(cnts[i], cnts_template[0], CONTOURS_MATCH_I2, 0);
        if (diff < 1){
            drawContours(wrench_draw, cnts, i, Scalar(0, 255, 0), 5);  
        }
        else
            drawContours(wrench_draw, cnts, i, Scalar(0, 0, 255), 5);  
    }
    imshow("wrench_draw", wrench_draw);
    waitKey();
}