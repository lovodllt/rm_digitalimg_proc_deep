cmake_minimum_required(VERSION 3.0.2)
project(rm_digitalimg_proc_deep)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
#set(OpenCV_DIR "/home/lovod/opencv-4.8.1/build/installed/lib/cmake/opencv4")
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        tf
        rm_vision
        message_generation
        dynamic_reconfigure
        std_msgs
        geometry_msgs
        visualization_msgs
        image_transport
        cv_bridge
        image_geometry
        nodelet
        tf2
        tf2_geometry_msgs
        rm_msgs
)

#find_package(OpenCV 4.8.1 EXACT REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCV 4 REQUIRED)

generate_dynamic_reconfigure_options(cfg/Inference.cfg)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES rm_digitalimg_proc_deep
        CATKIN_DEPENDS
        rm_vision
        roscpp
        tf
        message_generation
        dynamic_reconfigure
        std_msgs
        geometry_msgs
        visualization_msgs
        image_transport
        cv_bridge
        image_geometry
        nodelet
        tf2
        tf2_geometry_msgs
)

include_directories(
        ${CMAKE_INCLUDE_CURRENT_DIR}/include/
        ${PROJECT_SOURCE_DIR}/include/
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OpenVINO_INCLUDE_DIRS}
)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

add_library(rm_digitalimg_proc_deep
        src/inference.cpp
        src/number_classifier.cpp
        src/video_detect.cpp
        src/process_plugin.cpp
)

target_link_libraries(rm_digitalimg_proc_deep
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        openvino::runtime
)

add_executable(
        video_detect
        src/video_detect.cpp
        src/inference.cpp
        src/number_classifier.cpp
)

target_link_libraries(video_detect
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        openvino::runtime
)



