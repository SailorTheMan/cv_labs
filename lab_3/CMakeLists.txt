cmake_minimum_required(VERSION 3.2)
project(lab3)
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    )
add_executable(task1 src/task1.cpp)
add_executable(task2 src/task2.cpp)
add_executable(task3 src/task3.cpp)
add_executable(task4 src/task4.cpp)

target_link_libraries(task1 ${OpenCV_LIBS})
target_link_libraries(task2 ${OpenCV_LIBS})
target_link_libraries(task3 ${OpenCV_LIBS})
target_link_libraries(task4 ${OpenCV_LIBS})

