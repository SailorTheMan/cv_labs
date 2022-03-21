#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;

class Vector{
    public:
        Vector();
        Vector(float v1, float v2);
        float v1;
        float v2;
        float length;
        void normalize();
        Vector operator+(const Vector& accel);
        Vector operator-(const Vector& accel);
        Vector operator*(const float scalar);
};

Vector::Vector(){
    this->v1 = 0;
    this->v2 = 0;
    this->length = 0;
}

Vector::Vector(float v1, float v2){
    this->v1 = v1;
    this->v2 = v2;
    this->length = sqrt(pow(v1, 2) + pow(v2, 2));
}

void Vector::normalize(){
    v1 /= length;
    v2 /= length;
    length = 1;
}

Vector Vector::operator+(const Vector& accel){
    return Vector(this->v1 + accel.v1, this->v2 + accel.v2);
}

Vector Vector::operator-(const Vector& accel){
    return Vector(this->v1 - accel.v1, this->v2 - accel.v2);
}

Vector Vector::operator*(const float scalar){
    return Vector(this->v1 * scalar, this->v2 * scalar);
}

class Planet{
    public:
        Planet(Point2f center, int radius, int mass, Vector init_velocity, Vector accel, Scalar color);
        Scalar color;
        Vector accel;
        int mass;
        Point2f center;
        int radius;
        Vector velocity;
        void update_position();
};

Planet::Planet(Point2f center, int radius, int mass, Vector init_velocity, Vector accel, Scalar color){
    this->center.x = center.x;
    this->center.y = center.y;
    this->radius = radius;
    this->mass = mass;
    this->velocity = init_velocity;
    this->accel = accel;
    this->color = color;
}

void Planet::update_position(){
    velocity = velocity + accel;
    center.x += velocity.v1;
    center.y += velocity.v2;
}


int distance(Planet pl1, Planet pl2){
    return sqrt(pow(pl1.center.x - pl2.center.x, 2) + pow(pl1.center.y - pl2.center.y, 2));
}


class System{
    private:
        std::vector<Planet> pl_array;
    public:
        void update();
        void add_planet(Planet planet);
        std::vector<Planet> get_pl_array();
        void render_self(Mat img, bool draw_vels, bool draw_accels);
};

void System::add_planet(Planet planet){
    this->pl_array.push_back(planet);
}

std::vector<Planet> System::get_pl_array(){
    return this->pl_array;
}

void System::update(){
    for (size_t i = 0; i < pl_array.size(); i++)
    {
        Vector accel(0, 0);
        for (size_t j = 0; j < pl_array.size(); j++)
        {
            if (i == j) {continue;}
            float accel_val = 6.67 * pow(10, -3) * pl_array[i].mass * pl_array[j].mass / 
                                (pow( distance(pl_array[i], pl_array[j]), 2) * pl_array[i].mass);
            Vector inc(pl_array[j].center.x - pl_array[i].center.x, 
                      pl_array[j].center.y - pl_array[i].center.y);
            inc.normalize();
            inc = inc * accel_val;
            accel = accel + inc;
        }
        pl_array[i].accel = accel;
        pl_array[i].update_position();
    }
}

void System::render_self(Mat img, bool draw_vels = false, bool draw_accels = false){
    for (size_t i = 0; i < pl_array.size(); i++)
    {
        circle(img, pl_array[i].center, pl_array[i].radius, pl_array[i].color, -1);
        if (draw_vels){
            line(img, pl_array[i].center, 
            Point(pl_array[i].center.x + pl_array[i].velocity.v1,  pl_array[i].center.y + pl_array[i].velocity.v2), 
            Scalar(0, 0, 255), 2);
        }
        if (draw_accels){
            line(img, pl_array[i].center, 
            Point(pl_array[i].center.x + pl_array[i].accel.v1,  pl_array[i].center.y + pl_array[i].accel.v2), 
            Scalar(255, 0, 0), 2);
        }
    }
    
}

int main(int argc, char const *argv[])
{
    Mat bg = imread("../data/background.jpg");
    Planet sun(Point2f(bg.cols/2, bg.rows/2), 20, 1000000, Vector(0, 0), Vector(0, 0), Scalar(0, 100, 255));
    Planet earth(Point2f(bg.cols/2, 450), 20, 100, Vector(5, 0), Vector(0, 0), Scalar(255, 255, 0));
    Planet mars(Point2f(bg.cols/2, 100), 15, 150, Vector(-5, 0), Vector(0, 0), Scalar(0, 100, 0));
    System system;
    system.add_planet(sun);
    system.add_planet(mars);
    system.add_planet(earth);
    while (true)
    {
        system.update();
        Mat img = bg.clone();
        system.render_self(img, true, true);
        imshow("planets", img);
        waitKey(20);    
    }
    
    return 0;
}
