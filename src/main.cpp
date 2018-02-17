#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SIFT/SIFTDescriptor.h"

using namespace cv;
using namespace std;

const int SIZE = 27;
const int MID_POINT = SIZE/2;

Mat harrisCornerDetector(Mat input);
Mat nonMaximaSuppression(Mat input, Mat original);
Mat featureDescription(Mat original, Mat features);

vector<SIFTDescriptor> featureDescriptions = vector<SIFTDescriptor>();

int main() {
    //Mat image = imread("images/graf/img1.ppm", IMREAD_UNCHANGED);
    Mat image = imread("images/yosemite/Yosemite2.jpg", IMREAD_COLOR);
    //Mat image = imread("images/checkers.png", IMREAD_COLOR);
    Mat gray = Mat::zeros(image.size(), CV_32F);
    Mat output = Mat::zeros(image.size(), CV_32F);

    imshow("Original", image);
    cvtColor(image, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);
    gray *= 1./255;

    Mat h = harrisCornerDetector(gray);
    output = nonMaximaSuppression(h, image);

    imshow("Harris Corners", image);

    featureDescription(gray, output);

    waitKey(0);

    return 0;
}


/**
 * Detect feature points using Harris Corner Detection
 *
 * @param input a CV_32F gray scale matrix
 * @return Mat (CV_32F)
 */
Mat harrisCornerDetector(Mat input) {
    Mat Ix = Mat(input.size(), CV_32F);
    Mat Iy = Mat(input.size(), CV_32F);
    Mat Ixy = Mat(input.size(), CV_32F);
    Mat C = Mat::zeros(input.size(), CV_32F);          // C = det(H)/Trace(H)
    Mat result = Mat::zeros(input.size(), CV_32F);     // Final result with non-maximum suppression

    // Sobel Kernel
    float sobelFilter[] = {
            1.0/6, 0, -1.0/6,
            2.0/6, 0, -2.0/6,
            1.0/6, 0, -1.0/6
    };

    // Sobel Filters
    Mat s_v = Mat(3, 3, CV_32F, sobelFilter);
    Mat s_h = s_v.t();      // Transpose of horizontal Sobel matrix

    // Compute Ix, Iy, Ixy derivatives using Sobel
    filter2D(input, Ix, -1, s_h);
    filter2D(input, Iy, -1, s_v);

    // Create Harris matrix
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float x = Ix.at<float>(i,j);
            float y = Iy.at<float>(i,j);
            Ix.at<float>(i,j) = x * x;      // IxIx
            Iy.at<float>(i,j) = y * y;      // IxIy
            Ixy.at<float>(i,j) = x * y;     // IxIy
        }
    }

    // Apply Gaussian
    GaussianBlur(Ix, Ix, Size(3, 3), 1);
    GaussianBlur(Iy, Iy, Size(3, 3), 1);
    GaussianBlur(Ixy, Ixy, Size(3, 3), 1);

    // Calculate corner strength matrix
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float a = Ix.at<float>(i, j);
            float b = Ixy.at<float>(i, j);
            float c = Ixy.at<float>(i, j);
            float d = Iy.at<float>(i, j);
            float cornerStrength;


            // c(H) = det/trace, det = ad-bc, trace = a+d
            if (a + d == 0) {
                cornerStrength = 0;
            }
            else {
                cornerStrength = ((a*d)-(b*c))/(a + d);
            }

            // Apply threshold
            if (cornerStrength > 0.008) {
                C.at<float>(i, j) = cornerStrength;
            }
        }
    }

    waitKey(0);

    return C;
}

/**
 * Suppress non maximum values in a matrix of corner strength values for Harris Corner Detection
 *
 * @param input Partially complete Harris Corner matrix with corner strength values
 * @return Mat (CV_32F)
 */
Mat nonMaximaSuppression(Mat input, Mat original) {
    Mat suppressed = Mat(input.size(), CV_32F);
    float max = 0;
    float current;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = MID_POINT; i < input.rows - MID_POINT; i += SIZE) {
        for (int j = MID_POINT; j < input.cols - MID_POINT; j += SIZE) {
            max = 0;

            for (int k = i - MID_POINT; k <= i + MID_POINT; k++) {
                for (int l = j - MID_POINT; l <= j + MID_POINT; l++) {
                    current = input.at<float>(k, l);

                    if (current > max) {
                        max = current;
                        maxRow = k;
                        maxCol = l;
                    }
                }
            }

            if (max > 0) {
                suppressed.at<float>(maxRow, maxCol) = max;
                circle(original, Point(maxCol, maxRow), 4, Scalar(0, 0, 0, 255));
            }
        }
    }

    return suppressed;
}

/**
 * Describe features using SIFT descriptors for feature points
 *
 * @param original source image
 * @param features non-maximally suppressed harris matrix of features
 * @return
 */
Mat featureDescription(Mat original, Mat features) {
    //const int WINDOW_SIZE = 16;
    const int MID = WINDOW_SIZE/2;
    const int GRID_SIZE = 4;

    Mat gradientImage = Mat(original.size(), CV_32F);
    Mat Ix = Mat(original.size(), CV_32F);
    Mat Iy = Mat(original.size(), CV_32F);

    // Sobel Kernel
    float sobelFilter[] = {
            1.0/6, 0, -1.0/6,
            2.0/6, 0, -2.0/6,
            1.0/6, 0, -1.0/6
    };

    Mat s_v = Mat(3, 3, CV_32F, sobelFilter);
    Mat s_h = s_v.t();

    // Gradient Images
    filter2D(original, Ix, -1, s_h);
    filter2D(original, Iy, -1, s_v);

    for (int i = MID; i < original.rows - MID; i++) {
        for (int j = MID; j < original.cols - MID; j++) {
            float point = features.at<float>(i,j);

            if (point > 0) {
                float x, y;
                float windowX[WINDOW_SIZE][WINDOW_SIZE];
                float windowY[WINDOW_SIZE][WINDOW_SIZE];

                for (int k = i-MID; k < i+MID-1; k++) {
                    for (int l = j-MID; l < j+MID-1; l++) {
                        x = Ix.at<float>(k, l);
                        y = Iy.at<float>(k, l);
                        windowX[k-(i-MID)][l-(j-MID)] = x;
                        windowY[k-(i-MID)][l-(j-MID)] = y;
                    }
                }

                SIFTDescriptor d = SIFTDescriptor(windowX, windowY);
                d.generateHistograms();

                featureDescriptions.push_back(d);
            }
        }
    }

    return Mat();
}
