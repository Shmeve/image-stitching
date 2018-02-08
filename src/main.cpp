#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const int SIZE = 11;
const int MID_POINT = SIZE/2;

Mat harrisCornerDetector(Mat input);
Mat nonMaximaSuppression(Mat input);

int main() {
    //Mat image = imread("images/graf/img1.ppm", IMREAD_UNCHANGED);
    Mat image = imread("images/checkers.png", IMREAD_COLOR);
    Mat gray;
    Mat output;

    cvtColor(image, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    //output.convertTo(output, CV_8U);
    //imshow("Gray 32", output);

    Mat h = harrisCornerDetector(gray);
    output = nonMaximaSuppression(h);

    imshow("h", h);
    imshow("output", output);

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
    Mat C = Mat(input.size(), CV_32F);          // C = det(H)/Trace(H)
    Mat result = Mat(input.size(), CV_32F);     // Final result with non-maximum suppression

    // Setup Sobel filters to calculate the derivatives
    float sobelFilter[] = {
            1, 0, -1,
            2, 0, -2,
            1, 0, -1
    };

    Mat s_v = Mat(3, 3, CV_32F, sobelFilter);
    Mat s_h = s_v.t();      // Transpose of horizontal Sobel matrix

    // Compute Ix, Iy, Ixy
    filter2D(input, Ix, -1, s_h);
    filter2D(input, Iy, -1, s_v);
    multiply(Ix, Iy, Ixy);


    // Apply Gaussian
    GaussianBlur(Ix, Ix, Size(3, 3), 1);
    GaussianBlur(Iy, Iy, Size(3, 3), 1);
    GaussianBlur(Ixy, Ixy, Size(3, 3), 1);

    // Create Harris
    pow(Ix, 2, Ix); // Ix Ix
    pow(Iy, 2, Iy); // Iy Iy

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float a = Ix.at<float>(i, j);
            float b = Ixy.at<float>(i, j);
            float c = Ixy.at<float>(i, j);
            float d = Iy.at<float>(i, j);

            C.at<float>(i, j) = ((a * d) - (b * c))/(a + d);
        }
    }

    // Display/Test
    imshow("Ix", Ix);
    imshow("Iy", Iy);
    imshow("Ixy", Ixy);
    imshow("C", C);

    waitKey(0);

    return C;
}

/**
 * Suppress non maximum values in a matrix of corner strength values for Harris Corner Detection
 *
 * @param input Partially complete Harris Corner matrix with corner strength values
 * @return Mat (CV_32F)
 */
Mat nonMaximaSuppression(Mat input) {
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

            suppressed.at<float>(maxRow, maxCol) = max;
        }
    }

    return suppressed;
}
