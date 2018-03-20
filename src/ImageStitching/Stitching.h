#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Tools/Match.h"

class Stitching {
private:
    int bestInlierCount;
    cv::Mat bestHomography;
    cv::Mat finalHomography;
    cv::Mat image1;
    cv::Mat image2;
    std::vector<Match> matches;
    std::vector<Match> computerInlierCount(cv::Mat H, int inlierThreshold);
    Match getRandomMatch();
    bool pointInImage(cv::Point p, cv::Mat img);
public:
    cv::Point2f project(cv::Point2f p1, cv::Mat H);
    Stitching(std::string img1, std::string img2, std::vector<Match> matches);
    cv::Mat RANSAC(int numMatches, int numIterations, int inlierThreshold);
    cv::Mat drawMatches(int inlierThreshold, cv::Mat H);
    cv::Mat stitch(std::string writeTo);
    int getBestInlierCount() const;
    const cv::Mat &getBestHomography() const;
};