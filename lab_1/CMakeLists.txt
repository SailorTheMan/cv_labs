cmake_minimum_required(VERSION 3.2)
project(lab1)
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    )
add_executable(task src/task1.cpp)
add_executable(tank src/tank.cpp)
add_executable(planets src/planets.cpp)
add_executable(blending src/blending.cpp)
target_link_libraries(task ${OpenCV_LIBS})
target_link_libraries(tank ${OpenCV_LIBS})
target_link_libraries(planets ${OpenCV_LIBS})
target_link_libraries(blending ${OpenCV_LIBS})

