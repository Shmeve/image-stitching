#include "FeatureDetector_498.h"

using namespace cv;
using namespace std;

FeatureDetector_498::FeatureDetector_498(string file) {
    // Init original image
    this->raw = imread(file, IMREAD_COLOR);
    this->image = imread(file, IMREAD_COLOR);
    this->gray = Mat::zeros(image.size(), image.type());

    // Init gray scale image
    cvtColor(this->image, this->gray, COLOR_BGR2GRAY);
    this->gray.convertTo(this->gray, CV_32F);
    this->gray *= 1./255;

    // Init derivative images
    this->Ix = Mat::zeros(this->gray.size(), CV_32F);
    this->Iy = Mat::zeros(this->gray.size(), CV_32F);

    float sobelFilter[] = {
            1.0/6, 0, -1.0/6,
            2.0/6, 0, -2.0/6,
            1.0/6, 0, -1.0/6
    };

    // Sobel Filters
    Mat s_v = Mat(3, 3, CV_32F, sobelFilter);
    Mat s_h = s_v.t();      // Transpose of horizontal Sobel matrix

    filter2D(this->gray, this->Ix, -1, s_h);
    filter2D(this->gray, this->Iy, -1, s_v);

    //Init helper matrices
    this->harris = Mat::zeros(image.size(), CV_32F);
    this->suppressed = Mat::zeros(image.size(), CV_32F);

    // Init descriptor vector
    this->descriptors = vector<SIFTDescriptor>();
}

/**
 * Detect feature points using Harris Corner Detection
 *
 * @return Mat (CV_32F)
 */
Mat FeatureDetector_498::harrisCornerDetector() {
    Mat Ixx = Mat::zeros(Ix.size(), CV_32F);
    Mat Iyy = Mat::zeros(Iy.size(), CV_32F);
    Mat Ixy = Mat::zeros(Ix.size(), CV_32F);
    Mat Corners = Mat::zeros(image.size(), CV_32F);

    // Create Harris matrix components
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            float x = Ix.at<float>(i,j);
            float y = Iy.at<float>(i,j);
            Ixx.at<float>(i,j) = x * x;     // IxIx
            Iyy.at<float>(i,j) = y * y;     // IyIy
            Ixy.at<float>(i,j) = x * y;     // IxIy
        }
    }

    // Apply Gaussian
    GaussianBlur(Ixx, Ixx, Size(3, 3), 1);
    GaussianBlur(Iyy, Iyy, Size(3, 3), 1);
    GaussianBlur(Ixy, Ixy, Size(3, 3), 1);

    // Calculate corner strength matrix
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float a = Ixx.at<float>(i, j);
            float b = Ixy.at<float>(i, j);
            float c = Ixy.at<float>(i, j);
            float d = Iyy.at<float>(i, j);
            float cornerStrength;


            // c(H) = det/trace, det = ad-bc, trace = a+d
            if (a + d == 0) {
                cornerStrength = 0;
            }
            else {
                cornerStrength = ((a*d)-(b*c))/(a + d);
            }

            // Apply threshold
            if (cornerStrength > HARRIS_THRESHOLD) {
                Corners.at<float>(i, j) = cornerStrength;
            }
        }
    }

    return Corners;
}

/**
 * Suppress non maximum values in a matrix of corner strength values from Harris Corner Detection
 *
 * @param harrisCorners non-suppressed corner strength values from harris corner detector
 * @return Mat (CV_32F)
 */
Mat FeatureDetector_498::nonMaximaSuppression(Mat harrisCorners) {
    Mat s = Mat::zeros(image.size(), CV_32F);
    float max = 0;
    float current;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = SUPPRESSION_MID; i < gray.rows - SUPPRESSION_MID; i += SUPPRESSION_WINDOW) {
        for (int j = SUPPRESSION_MID; j < gray.cols - SUPPRESSION_MID; j += SUPPRESSION_WINDOW) {
            max = 0;

            for (int k = i - SUPPRESSION_MID; k <= i + SUPPRESSION_MID; k++) {
                for (int l = j - SUPPRESSION_MID; l <= j + SUPPRESSION_MID; l++) {
                    current = harrisCorners.at<float>(k, l);

                    if (current > max) {
                        max = current;
                        maxRow = k;
                        maxCol = l;
                    }
                }
            }

            if (max > 0) {
                s.at<float>(maxRow, maxCol) = max;
                circle(image, Point(maxCol, maxRow), 4, Scalar(0, 0, 0, 255));
            }
        }
    }

    return s;
}

/**
 * Driver method for feature detection using Harris corner detection and non-maximal suppression
 *
 * @return Mat (CV_8U)
 */
Mat FeatureDetector_498::detectFeatures() {
    harris = harrisCornerDetector();
    suppressed = nonMaximaSuppression(harris);

    imshow("Original", raw);
    imshow("Image", image);

    waitKey(0);

    return image;
}

/**
 * Creates descriptions of feature interest points produced by harris corner detection
 *
 * @return vector<SIFTDescriptor>
 */
vector<SIFTDescriptor> FeatureDetector_498::describeFeatures() {
    imshow("suppressed", suppressed);
    int count = 0;
    for (int i = DESCRIPTOR_MID; i < suppressed.rows - DESCRIPTOR_MID; i++) {
        for (int j = DESCRIPTOR_MID; j < suppressed.cols - DESCRIPTOR_MID; j++) {
            float point = suppressed.at<float>(i,j);

            if (point > 0.0) {
                float x, y;
                float windowX[DESCRIPTOR_WINDOW][DESCRIPTOR_WINDOW];
                float windowY[DESCRIPTOR_WINDOW][DESCRIPTOR_WINDOW];

                for (int k = i-DESCRIPTOR_MID; k < i+DESCRIPTOR_MID; k++) {
                    for (int l = j-DESCRIPTOR_MID; l < j+DESCRIPTOR_MID; l++) {
                        x = Ix.at<float>(k, l);
                        y = Iy.at<float>(k, l);
                        windowX[k-(i-DESCRIPTOR_MID)][l-(j-DESCRIPTOR_MID)] = x;
                        windowY[k-(i-DESCRIPTOR_MID)][l-(j-DESCRIPTOR_MID)] = y;
                    }
                }

                SIFTDescriptor d = SIFTDescriptor(windowX, windowY);
                d.generateHistograms();
                count++;

                descriptors.push_back(d);
            }
        }
    }

    cout << count << endl;

    return descriptors;
}


