#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SIFT/SIFTDescriptor.h"
#include "FeatureDetector_498.h"

using namespace cv;
using namespace std;

int main() {
//    string img = "pano1_0008.tga";
//    string img2 = "images/panorama/pano1_0009.tga";
//    string img = "images/graf/img1.ppm";
//    string img2 = "images/graf/img2.ppm";
    string img = "images/yosemite/Yosemite1.jpg";
    string img2 = "images/yosemite/Yosemite2.jpg";
//    string img = "images/checkers.png";

    // Image 1
    FeatureDetector_498 fd = FeatureDetector_498(img);
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
            if (ssd < 3) {
                // TODO: Update best matches
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
                Point(bestDescriptor->getFeatureCol()+h.cols, bestDescriptor->getFeatureRow()), Scalar(0, 0, 0, 255));
            }
        }
    }

    imshow("Matching Points", matches);

    waitKey(0);

    return 0;
}