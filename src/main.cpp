#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "FeatureMatching/SIFT/SIFTDescriptor.h"
#include "FeatureMatching/FeatureDetector_498.h"
#include "Tools/Match.h"
#include "ImageStitching/Stitching.h"

using namespace cv;
using namespace std;

vector<Match> matchFeatures(string img1, string img2, string writeTo);

int main() {
    string img1 = "images/rainier/Rainier1.png";
    string img2 = "images/rainier/Rainier2.png";
    string img3 = "images/rainier/Rainier3.png";
    string img4 = "images/rainier/Rainier4.png";
    string img5 = "images/rainier/Rainier5.png";
    string img6 = "images/rainier/Rainier6.png";

    vector<Match> m1 = matchFeatures(img1, img2, "results/1.png");
    /*matchFeatures(img2, img3, "results/2.png");
    matchFeatures(img3, img4, "results/3.png");
    matchFeatures(img4, img5, "results/4.png");
    matchFeatures(img5, img6, "results/5.png");*/

    Stitching s = Stitching(img1, img2, m1);
    Mat m = s.RANSAC(m1.size(), 10, 15);
    cout << s.getBestInlierCount();

    imshow("res", m);
    waitKey(0);

    return 0;
}

/**
 * Detect interest points within two images and link matching pairs
 *
 * @param img1 Starting image to detect interest points
 * @param img2 Corresponding image to detect matching interest points
 * @return Mat containing both inputs with highlighted interest points and matches
 */
vector<Match> matchFeatures(string img1, string img2, string writeTo) {
    const double DISTANCE_THRESHOLD = 3.0;
    vector<Match> matchesList = vector<Match>();

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
                Match m = Match(descriptions.at(i), *bestDescriptor);
                matchesList.push_back(m);

                line(matches, Point(descriptions.at(i).getFeatureCol(), descriptions.at(i).getFeatureRow()),
                     Point(bestDescriptor->getFeatureCol()+h.cols, bestDescriptor->getFeatureRow()),
                     Scalar(0, 175, 0, 255), 1);
            }
        }
    }

    imshow("Matching Points", matches);
    imwrite(writeTo, matches);
    waitKey(0);

    return matchesList;
}