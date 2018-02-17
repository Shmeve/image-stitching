#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "SIFT/SIFTDescriptor.h"

class FeatureDetector_498 {
private:
    const static int SUPPRESSION_WINDOW = 27;
    const static int SUPPRESSION_MID = SUPPRESSION_WINDOW/2;
    constexpr static float HARRIS_THRESHOLD = 0.008 ;
    const static int DESCRIPTOR_WINDOW = 16;
    const static int DESCRIPTOR_MID = DESCRIPTOR_WINDOW/2;
    const static int GRID_SIZE = 4;
    cv::Mat raw;
    cv::Mat image;
    cv::Mat gray;
    cv::Mat harris;
    cv::Mat suppressed;
    cv::Mat Ix;
    cv::Mat Iy;
    std::vector<SIFTDescriptor> descriptors;

    cv::Mat harrisCornerDetector();
    cv::Mat nonMaximaSuppression(cv::Mat harrisCorners);
    void generateFeatureDescriptions();
public:
    FeatureDetector_498(std::string file);
    cv::Mat detectFeatures();
    std::vector<SIFTDescriptor> describeFeatures();
};