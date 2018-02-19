#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SIFT/SIFTDescriptor.h"
#include "FeatureDetector_498.h"

using namespace cv;
using namespace std;

int main() {
    //string img = "images/graf/img1.ppm";
    string img = "images/yosemite/Yosemite1.jpg";
    string img2 = "images/yosemite/Yosemite2.jpg";
    //string img = "images/checkers.png";

    // Image 1
    FeatureDetector_498 fd = FeatureDetector_498(img);
    fd.detectFeatures();
    vector<SIFTDescriptor> descriptions = fd.describeFeatures();

    // Image 2
    FeatureDetector_498 fd2 = FeatureDetector_498(img2);
    fd2.detectFeatures();
    vector<SIFTDescriptor> descriptions2 = fd2.describeFeatures();


    // Match?
    for (int i = 0; i < descriptions.size(); i++) {
        for (int j = 0; j < descriptions2.size(); j++) {
            cout << descriptions2.at(j).SSD(descriptions.at(i)) << endl;
        }
    }


    return 0;
}