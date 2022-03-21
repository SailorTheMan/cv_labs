#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>

using namespace cv;

void blur_roi(Mat *src, Mat *dst){
    const int channels = src->channels();
    if (channels != 1) {return;}
    int intensity = 0;
    for (size_t x = 0; x < src->cols; x++)
    {
        for (size_t y = 0; y < src->rows; y++)
        {
            intensity += (uchar)src->at<uchar>(y, x);
        }
        
    }
    intensity /= (src->cols * src->rows);
    dst->at<uchar>(dst->cols / 2, dst->rows / 2) = (uchar)intensity;
}

Mat box_filter(Mat src, Size kernel){
    Mat dst = src.clone();
    for (size_t i = 0; i < dst.cols - kernel.width; i ++)
    {
        for (size_t j = 0; j < dst.rows - kernel.height; j ++)
        {
            Rect rp(i, j, kernel.width, kernel.height);
            Mat dst_roi = dst(rp);
            Mat src_roi = src(rp);
            blur_roi(&src_roi, &dst_roi);
        }   
    }
    return dst;
}

Mat find_diffs(Mat img1, Mat img2){
    Mat dif_img = img2.clone();
    std::vector<float> diffs;
    for (size_t i = 0; i < img1.cols; i++)
    {
        for (size_t j = 0; j < img1.rows; j++)
        {
            float diff = abs((int)img1.at<uchar>(j, i) - (int)img2.at<uchar>(j,i));
            diffs.push_back(diff);
            dif_img.at<uchar>(j,i) = (uint)diff;
        }
        
    }
    float disparity = 0;
    for (size_t i = 0; i < diffs.size(); i++)
    {
        disparity += (diffs[i]/255) * 100;
    }
    std::cout << "Similarity: " << 100 - disparity/ diffs.size() << std::endl; 
    return(dif_img);
}

void logTransform(Mat& img, double c)
{
	
	MatIterator_<uint8_t> it, end;
	for (it = img.begin<uint8_t>(), end = img.end<uint8_t>(); it != end; ++it)
	{
		*it = (uint8_t)(c * log(*it + 1));
	}
}

void unsharpMask(Mat& original, Mat& blurred, Mat& output, double k = 1)
{
	output = original - blurred;
	output = k * output + original;
}

void laplaseFilter(cv::Mat& original)
{
	float kernelArray[] = {0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0};
	cv::Mat kernel(3, 3, CV_32FC1, kernelArray);

	cv::filter2D(original, original, -1, kernel, cv::Point(-1, -1));
}

void laplaseUnsharp(cv::Mat& original, cv::Mat& laplase, double k = 1)
{
	original = original - k * laplase;
}


int main(int argc, char const *argv[])
{
    int task = 3;


    Mat img = imread("../data/lena.png");
    cvtColor(img, img, CV_RGB2GRAY);
    if (task == 3){
        TickMeter tm;
        Mat blur_img;
        tm.start();
        blur(img, blur_img, Size(3, 3));
        tm.stop();
        std::cout<< "cv implementation: " << tm.getTimeMilli()<<std::endl;
        tm.start();
        Mat box_fil_img = box_filter(img, Size(3, 3));
        tm.stop();
        std::cout<< "own implementation: " << tm.getTimeMilli()<<std::endl;
        imshow("orig", img);
        imshow("blur", blur_img);
        imshow("box_filter", box_fil_img);
        Mat diff_img = find_diffs(blur_img, box_fil_img);
        imshow("diff", diff_img);
    }
    else if (task == 4){
        Mat blur_img = box_filter(img, Size(3, 3));
        Mat gauss_img;
        GaussianBlur(img, gauss_img, Size(3, 3), 0.8);
        Mat diff_img = find_diffs(blur_img, gauss_img);
        logTransform(diff_img, 10);
        imshow("box", blur_img);
        imshow("gauss", gauss_img);
        imshow("diff", diff_img);

    }
    else if (task == 5){
        // gaussFilter(blurCopy, 1.4, 7);
        Mat gauss_img;
		GaussianBlur(img, gauss_img, Size(7, 7), 1.4);
        Mat unsharped_gauss(img.rows, img.cols, img.type());
        Mat unsharped_box(img.rows, img.cols, img.type());
		unsharpMask(img, gauss_img, unsharped_gauss, 3);
		Mat blur_img = box_filter(img, Size(7, 7));
		unsharpMask(img, blur_img, unsharped_box, 3);
		Mat diff_img = find_diffs(unsharped_gauss, unsharped_box);
        imshow("img", img);
        imshow("unsharped_gauss", unsharped_gauss);
        imshow("unsharped_box", unsharped_box);
        imshow("diff", diff_img);
    }
    else if (task == 6){
        Mat laplase_img = img.clone();
        laplaseFilter(laplase_img);
        imshow("laplase_img", laplase_img);
    }
    else if (task == 7){
        Mat laplase_img = img.clone();
        laplaseFilter(laplase_img);
        Mat unsharp_laplase = img.clone();
        laplaseUnsharp(unsharp_laplase, laplase_img, 1.5);
        imshow("orig", img);
        imshow("laplase_img", laplase_img);
        imshow("unsharp", unsharp_laplase);
    }
    waitKey(0);
    return 0;
}
