
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp> 

#include <iostream>

using namespace cv;

Mat blend(Mat img1, Mat img2, double alp)
{

  
  resize(img2, img2, Size(600, 600), INTER_LINEAR);
  resize(img1, img1, Size(600, 600), INTER_LINEAR);
  Mat sum;
  sum.create(img1.rows, img1.cols, CV_8UC3);
  int x = 0;
  int y = 0;
  Vec3b& color3 = img1.at<Vec3b>(x,y);
  for (int y = 0; y < img1.rows; y++)
  {
    for (int x = 0; x < img1.cols; x++)
    {
      Vec3b& color1 = img1.at<Vec3b>(x,y);
      Vec3b& color2 = img2.at<Vec3b>(y,x);
      for (int i = 0; i <= 2; i++)
      {
        color3[i] = color1[i] * alp + color2[i] * (1-alp);
        
      }
      sum.at<Vec3b>(Point(x,y)) = color3;
    }
  }
  return sum;
}


int main( int argc, char** argv ) 
{
  
  
  Mat image1 = imread("../data/background.jpg");
  Mat image2 = imread("../data/tank.png");
  
  
  Mat out = blend(image1, image2, 0.5);
  double alpha = 0.5;
  while (1)
  {
    int key = waitKey(0);
    std::cout << alpha << std::endl;
    if (key == 83)
    {
      alpha += 0.01;
      if (alpha > 1.0) alpha = 1.0;
    }
    if (key == 81)
    {
      alpha -= 0.01;
      if (alpha < 0.0) alpha = 0.0;
    }
    if (key == 27) break;
    out = blend(image1, image2, alpha);
    imshow("out", out);
  }
  destroyAllWindows();
  return 0;
}