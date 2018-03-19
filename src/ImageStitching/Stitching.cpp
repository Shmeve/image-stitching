#include <stdlib.h>
#include "Stitching.h"

using namespace std;
using namespace cv;

Stitching::Stitching(string img1, string img2, vector<Match> matches) {
    this->bestInlierCount = 0;
    this->bestHomography = Mat::zeros(3, 3, 0);
    this->image1 = imread(img1, IMREAD_UNCHANGED);
    this->image2 = imread(img2, IMREAD_UNCHANGED);
    this->matches = matches;
}

Mat Stitching::RANSAC(int numMatches, int numIterations, int inlierThreshold) {
    int matchesPerIterations = 4;
    vector<Match> inliers = vector<Match>();

    for (int i = 0; i < numIterations; i++) {
        vector<Point2f> img1Points;   // src
        vector<Point2f> img2Points;   // dst

        // Generate Arrays
        for (int j = 0; j < matchesPerIterations; j++) {
            Match m = getRandomMatch();
            img1Points.emplace_back(Point_<float>(m.getPoint1().getFeatureRow(), m.getPoint1().getFeatureCol()));
            img2Points.emplace_back(Point_<float>(m.getPoint2().getFeatureRow(), m.getPoint2().getFeatureCol()));
        }

        Mat H = findHomography(img1Points, img2Points, 0);
        inliers = computerInlierCount(H, inlierThreshold);

        if (inliers.size() > this->bestInlierCount) {
            this->bestInlierCount = (int) inliers.size();
            this->bestHomography = H;
        }
    }

    return drawMatches(inlierThreshold);

    //return Mat::zeros(1,1,0);
}

Point2f Stitching::project(Point2f p1, Mat H) {
    Mat point = Mat::zeros(3, 1, CV_32F);
    point.at<float>(0,0) = p1.x;
    point.at<float>(1,0) = p1.y;
    point.at<float>(2,0) = 1.0;

    Mat projection;
    H.convertTo(projection, CV_32F);

    Mat result = projection * point;

    return Point2f(result.at<float>(0,0), result.at<float>(1,0));
}

vector<Match> Stitching::computerInlierCount(Mat H, int inlierThreshold) {
    int count = 0;
    vector<Match> inliers = vector<Match>();

    for (auto match : this->matches) {
        Point2f p1 = Point_<float>(match.getPoint1().getFeatureRow(), match.getPoint1().getFeatureCol());
        Point2f p2 = Point_<float>(match.getPoint2().getFeatureRow(), match.getPoint2().getFeatureCol());
        Point2f projection = project(p1, H);

        double distance = sqrt(pow(projection.x-p2.x, 2) + pow(projection.y-p2.y, 2));

        if (distance <= inlierThreshold) {
            count++;
            inliers.emplace_back(match);
        }
    }

    return inliers;
}

Mat Stitching::drawMatches(int inlierThreshold) {
    RNG rng(12345);
    vector<Match> inliers = computerInlierCount(this->bestHomography, inlierThreshold);
    int width = this->image1.cols + this->image2.cols;
    int height = (this->image1.rows > this->image2.rows) ? this->image1.rows : this->image2.rows;
    Mat result = Mat::zeros(height, width, this->image1.type());

    hconcat(this->image1, this->image2, result);

    for (auto match : inliers) {
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        Point p1 = Point(match.getPoint1().getFeatureCol(), match.getPoint1().getFeatureRow());
        Point p2 = Point(match.getPoint2().getFeatureCol()+this->image1.cols, match.getPoint2().getFeatureRow());

        circle(result, p1, 4, color);
        circle(result, p2, 4, color);
        line(result, p1, p2, color, 1);
    }

    return result;
}

/**
 * Grab random set of potentially matching points
 *
 * @return Match
 */
Match Stitching::getRandomMatch() {
    return this->matches.at((rand() % this->matches.size()-1) + 1);
}

int Stitching::getBestInlierCount() const {
    return bestInlierCount;
}

const Mat &Stitching::getBestHomography() const {
    return bestHomography;
}
