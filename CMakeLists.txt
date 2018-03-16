cmake_minimum_required(VERSION 3.5)
project(feature_detection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
set(SOURCE_FILES src/main.cpp src/FeatureMatching/SIFT/SIFTDescriptor.cpp src/FeatureMatching/SIFT/SIFTDescriptor.h src/FeatureMatching/FeatureDetector_498.cpp src/FeatureMatching/FeatureDetector_498.h src/ImageStitching/Stitching.cpp src/ImageStitching/Stitching.h src/Tools/Match.cpp src/Tools/Match.h)
add_executable(feature_detection ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(feature_detection ${OpenCV_LIBS})