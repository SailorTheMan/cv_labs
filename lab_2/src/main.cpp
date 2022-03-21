#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

void blurPixel(Mat* newImg, int x, int y);
void boxFiltering(Mat* img);
void imageDifferent(Mat orig, Mat blr);

Mat blurImg;
Mat imDiff;

int main(int argc, char** argv ) { 
    
    Mat origImage;
    origImage = imread("../data/lena.png");

    Mat grayImage;
    cvtColor(origImage, grayImage, COLOR_BGR2GRAY);
    imshow("grayLenna.png", grayImage);

    boxFiltering(&grayImage);
    imshow("blurLenna.png", blurImg);

    Mat opencvBlurImage;
    blur(grayImage,opencvBlurImage,Size(3,3),Point(-1,-1));   
    imshow("open.png", opencvBlurImage);

    imageDifferent(blurImg, opencvBlurImage);
    imshow("dif.png", imDiff);
    waitKey(0);

    return 0;
}

void blurPixel(Mat* newImg, int x, int y) {
    int sum = 0;
    int rCnt, colCnt;
    for(colCnt = x; colCnt < (x + 3); colCnt++) {
        
        for (rCnt = y; rCnt < (y + 3); rCnt++) {
            sum += newImg->at<uchar>(rCnt, colCnt);
        }
    }
    newImg->at<uchar>(y + 1, x + 1) = (sum / 9);
}

void boxFiltering(Mat* img) {
    blurImg = img->clone();
    int iRows = img->rows;
    int iCols = img->cols;
    int i,j;
    for( i = 0; i < (iRows - 3); i++) {
        for ( j = 0; j < (iCols - 3); j++) {
            blurPixel(&blurImg, i, j);
        }
    }
}

void imageDifferent(Mat orig, Mat blr){
    float dif;
    float sumDif = 0;
    imDiff = blurImg.clone();
    for( int l = 0; l < orig.rows; l++) {
        for (int m = 0; m < orig.cols; m++) {
            dif = abs((orig.at<uchar>(l,m) - blr.at<uchar>(l,m)));
            sumDif += dif;
            imDiff.at<uchar>(l,m) = (uint)dif;
        }
    }
    float differ, sim;
    differ = (sumDif / 255) * 100;
    sim = 100 - (differ/ (blurImg.cols * blurImg.rows));
    std::cout<< "Сходство: "<<  sim << std::endl;

}