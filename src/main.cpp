#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const int SIZE = 7;
const int MID_POINT = SIZE/2;

Mat harrisCornerDetector(Mat input);
Mat nonMaximaSuppression(Mat input);

int main() {
    //Mat image = imread("images/graf/img1.ppm", IMREAD_UNCHANGED);
    Mat image = imread("images/yosemite/Yosemite2.jpg", IMREAD_COLOR);
    //Mat image = imread("images/checkers.png", IMREAD_COLOR);
    Mat gray = Mat::zeros(image.size(), CV_32F);
    Mat output = Mat::zeros(image.size(), CV_32F);;

    imshow("Original", image);
    image.convertTo(image, CV_32F);
    cvtColor(image, gray, COLOR_BGR2GRAY);

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

    // Sobel Kernel
    float sobelFilter[] = {
            1.0/8, 0, -1.0/8,
            2.0/8, 0, -2.0/8,
            1.0/8, 0, -1.0/8
    };

    // Sobel Filters
    Mat s_v = Mat(3, 3, CV_32F, sobelFilter);
    Mat s_h = s_v.t();      // Transpose of horizontal Sobel matrix

    // Compute Ix, Iy, Ixy derivatives using Sobel
    filter2D(input, Ix, -1, s_h);
    filter2D(input, Iy, -1, s_v);
    multiply(Ix, Iy, Ixy);


    // Apply Gaussian
    GaussianBlur(Ix, Ix, Size(3, 3), 1);
    GaussianBlur(Iy, Iy, Size(3, 3), 1);
    GaussianBlur(Ixy, Ixy, Size(3, 3), 1);

    // Create Harris matrix
    pow(Ix, 2, Ix); // Ix Ix
    pow(Iy, 2, Iy); // Iy Iy

    // Compute sum of products of the derivatives
    Mat IxSum;
    Mat IySum;
    Mat IxySum;
    float sumKernel[] = {
            1.0/9, 1.0/9, 1.0/9,
            1.0/9, 1.0/9, 1.0/9,
            1.0/9, 1.0/9, 1.0/9
    };

    Mat sumOfProducts = Mat(3, 3, CV_32F, sumKernel);

    filter2D(Ix, IxSum, -1, sumOfProducts);
    filter2D(Iy, IySum, -1, sumOfProducts);
    filter2D(Ixy, IxySum, -1, sumOfProducts);

    // Calculate corner strength matrix
    for (int i = MID_POINT; i < input.rows - MID_POINT; i++) {
        for (int j = MID_POINT; j < input.cols - MID_POINT; j++) {
            float a = IxSum.at<float>(i, j);
            float b = IxySum.at<float>(i, j);
            float c = IxySum.at<float>(i, j);
            float d = IySum.at<float>(i, j);

            // c(H) = det/trace, det = ad-bc, trace = a+d
            float cornerStrength = (a*d - b*c)/(a + d);

            // Apply threshold
            if (cornerStrength > 10) {
                C.at<float>(i, j) = cornerStrength;
            }
        }
    }

    // Display/Test
    /*imshow("Ix", Ix);
    imshow("Iy", Iy);
    imshow("Ixy", Ixy);
    imshow("C", C);*/

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
