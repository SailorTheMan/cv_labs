cmake_minimum_required(VERSION 3.2)
project(lab5)
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    )
add_executable(task1 main.cpp)

target_link_libraries(task1 ${OpenCV_LIBS})

