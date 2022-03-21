cmake_minimum_required(VERSION 3.0)
project(lab2)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(box_filter src/box_filter.cpp)
target_link_libraries(box_filter ${OpenCV_LIBS})

add_executable(diana src/main.cpp)
target_link_libraries(diana ${OpenCV_LIBS})