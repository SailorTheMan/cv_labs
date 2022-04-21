cmake_minimum_required(VERSION 3.2)
project(lab4)
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    )
add_executable(task1 src/task1.cpp)
target_link_libraries(task1 ${OpenCV_LIBS})
