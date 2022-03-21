#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;

void draw_rotated_rectangle(Mat img, RotatedRect rect){
    Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(img, vertices[i], vertices[(i+1)%4], Scalar(0, 255, 0), 2);
    }
    
}

float get_angle(Point2f prev_pt, Point2f cur_pt){
    float dx = cur_pt.x - prev_pt.x;
    float dy = prev_pt.y - cur_pt.y;
    float angle = atan2(dy, dx);
    return angle;

}

int main(int argc, char const *argv[])
{
    Mat bg = imread("../data/background.jpg");
    
    float prev_x = 0;
    float prev_y = bg.rows/2;
    bool have_shot = false;
    for (int x = 0; x < 1000; x +=5)
    {
        Mat img = bg.clone();
        float y = sin(5*x) * (img.rows/2 - 20) + img.rows / 2; // scale and shift sin to fit the screen
        line(bg, Point2f(prev_x, prev_y), Point2f(x, y), Scalar(0, 255, 0));
        float angle = atan2(y - prev_y, x-prev_x);
        draw_rotated_rectangle(img, RotatedRect(Point2f(x, y), Size2f(40, 20), angle*180/3.14));
        prev_x = x;
        prev_y = y;
        // rectangle(img, Rect(img.cols/2 - 10, img.rows/2 - 10, 20, 20), Scalar(0, 0, 255));
        imshow("task1", img); 
        waitKey(50);
        // if the cart is a 20 pixels away from the center of the picture or closer
        if ((sqrt(pow(img.cols/2 - x, 2) + pow(img.rows/2 - y, 2)) < 20) && ! have_shot){
            imwrite("../data/output.png", img);
            have_shot = true;
        }
        
    }

}