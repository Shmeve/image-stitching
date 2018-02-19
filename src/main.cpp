#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SIFT/SIFTDescriptor.h"
#include "FeatureDetector_498.h"

using namespace cv;
using namespace std;

Mat matchFeatures(string img1, string img2);

int main() {
    string img1 = "images/yosemite/Yosemite1.jpg";
    string img2 = "images/yosemite/Yosemite2.jpg";
    string img3 = "images/graf/img1.ppm";
    string img4 = "images/graf/img2.ppm";
    string img5 = "images/graf/img4.ppm";

    imwrite("results/1.png", matchFeatures(img1, img2));
    imwrite("results/2.png", matchFeatures(img2, img1));
    imwrite("results/3.png", matchFeatures(img3, img4));
    imwrite("results/4.png", matchFeatures(img3, img5));

    return 0;
}

/**
 * Detect interest points within two images and link matching pairs
 *
 * @param img1 Starting image to detect interest points
 * @param img2 Corresponding image to detect matching interest points
 * @return Mat containing both inputs with highlighted interest points and matches
 */
Mat matchFeatures(string img1, string img2) {
    const double DISTANCE_THRESHOLD = 3.0;

    // Image 1
    FeatureDetector_498 fd = FeatureDetector_498(img1);
    Mat h = fd.detectFeatures();
    vector<SIFTDescriptor> descriptions = fd.describeFeatures();

    // Image 2
    FeatureDetector_498 fd2 = FeatureDetector_498(img2);
    Mat h2 = fd2.detectFeatures();
    vector<SIFTDescriptor> descriptions2 = fd2.describeFeatures();

    // Concatenate two images for final display
    Mat matches = Mat::zeros(h.rows, h.cols+h2.cols, h.type());
    hconcat(h, h2, matches);

    // Match
    for (size_t i = 0; i < descriptions.size(); i++) {
        SIFTDescriptor *bestDescriptor, *secondBestDescriptor;
        double bestMatch = DBL_MAX;
        double secondBestMatch = DBL_MAX;

        for (size_t j = 0; j < descriptions2.size(); j++) {
            double ssd = descriptions2.at(j).SSD(descriptions.at(i));
            if (ssd < DISTANCE_THRESHOLD) {
                if (ssd < bestMatch) {
                    secondBestMatch = bestMatch;
                    bestMatch = ssd;
                    secondBestDescriptor = bestDescriptor;
                    bestDescriptor = &descriptions2.at(j);
                }
                else if (ssd < secondBestMatch) {
                    secondBestMatch = ssd;
                    secondBestDescriptor = &descriptions2.at(j);
                }
            }
        }

        // Found a match
        if (bestMatch > 0) {
            // Calculate ratio of two best matches
            double ratioScore = bestMatch/secondBestMatch;

            // Match is unambiguous
            if (ratioScore < 0.005) {
                line(matches, Point(descriptions.at(i).getFeatureCol(), descriptions.at(i).getFeatureRow()),
                     Point(bestDescriptor->getFeatureCol()+h.cols, bestDescriptor->getFeatureRow()),
                     Scalar(0, 175, 0, 255), 1);
            }
        }
    }

    imshow("Matching Points", matches);
    waitKey(0);

    return matches;
}