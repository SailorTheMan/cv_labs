#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;


Mat3b draw_tank(Mat3b img, Mat4b tank, Point pos)
{
  double alpha = 1; // alpha in [0,1]

  Mat3b roi = img(Rect(pos.x, pos.y, tank.cols, tank.rows));

  for (int r = 0; r < roi.rows; ++r)
  {
    for (int c = 0; c < roi.cols; ++c)
    {
      const Vec4b& vf = tank(r,c);
      if (vf[3] > 0) // alpha channel > 0
      {
        // Blending
        Vec3b& vb = roi(r,c);
        vb[0] = alpha * vf[0] + (1 - alpha) * vb[0];
        vb[1] = alpha * vf[1] + (1 - alpha) * vb[1];
        vb[2] = alpha * vf[2] + (1 - alpha) * vb[2];
      }
    }
  }

  return img;
}


class Bullet{
    public:
    Bullet();
    Bullet(Point2i center, int dir);
    Point2i center;
    int direction;
    void update_position();
};

Bullet::Bullet(Point2i center, int dir){
    this->center = center;
    this->direction = dir;
}

void Bullet::update_position(){
    switch(direction)
    {
        case -1:
            center.y -= 10;
            break;
        case ROTATE_180:
            center.y += 10;
            break;
        case ROTATE_90_CLOCKWISE:
            center.x += 10;
            break;
        case ROTATE_90_COUNTERCLOCKWISE:
            center.x -= 10;
            break;
    }
}


int main(int argc, char const *argv[])
{
    Mat bg = imread("../data/background.jpg");
    Mat tank_big = imread("../data/tank.png");
    Mat tank_lil;

    resize(tank_big, tank_lil, Size2f(tank_big.rows/3, tank_big.cols/3));
    Point2i tank_center(200, 200);
    int rotate_dir = -1;
    std::vector<Bullet> bl_array;
    while (true)
    {
        Mat img = bg.clone();
        Mat tank = tank_lil.clone();
        switch ((char)waitKey(50))
        {
            case 119: // `w` is pressed, moving upward
                tank_center.y -= 5;
                rotate_dir = -1;
                break;
            case 115: // `s`
                tank_center.y += 5;
                rotate_dir = ROTATE_180;
                break;
            case 97: // `a`
                tank_center.x -= 5;
                rotate_dir = ROTATE_90_COUNTERCLOCKWISE;
                break;
            case 100: // `d`
                tank_center.x += 5;
                rotate_dir = ROTATE_90_CLOCKWISE;
                break;
            case 99: // `c`
            {
                Point2i bl_center(tank_center.x + tank.cols/2,
                                    tank_center.y + tank.rows/2);
                Bullet bullet(bl_center, rotate_dir);
                bl_array.push_back(bullet);
                break;
            }
            default:
                break;
        }
        if (rotate_dir != -1){
            rotate(tank_lil, tank, rotate_dir);
        }
        draw_tank(img, tank, tank_center);
        for (size_t i = 0; i < bl_array.size(); i++)
            {
                if((bl_array[i].center.x > img.cols) || (bl_array[i].center.y > img.rows) ||
                    (bl_array[i].center.x < 0) || (bl_array[i].center.y < 0)){
                        bl_array.erase(bl_array.begin()+i);
                    }
                bl_array[i].update_position();
                circle(img, bl_array[i].center, 10, Scalar(0, 100, 255), -1);
            }
        imshow("tank", img);
        
    }
    
    return 0;
}
