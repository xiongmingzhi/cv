cmake_minimum_required(VERSION 3.16)
project(cv)
set(CMAKE_CXX_STANDARD 11)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "D:/opencv-4.1.1/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn)

add_executable(cv main.cpp)
add_executable(sober sober.cpp)
add_executable(practice practice.cpp)
add_executable(test test.cpp)
# linking
target_link_libraries(cv ${OpenCV_LIBS})
target_link_libraries(practice ${OpenCV_LIBS})
target_link_libraries(test ${OpenCV_LIBS})
