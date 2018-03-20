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

    // Estimate Homography using 4 random matches
    for (int i = 0; i < numIterations; i++) {
        vector<Point2f> img1Points;   // src
        vector<Point2f> img2Points;   // dst

        // Generate Arrays
        for (int j = 0; j < matchesPerIterations; j++) {
            Match m = getRandomMatch();
            img1Points.emplace_back(Point_<float>(m.getPoint1().getFeatureCol(), m.getPoint1().getFeatureRow()));
            img2Points.emplace_back(Point_<float>(m.getPoint2().getFeatureCol(), m.getPoint2().getFeatureRow()));
        }

        Mat H = findHomography(img1Points, img2Points, 0);
        inliers = computerInlierCount(H, inlierThreshold);

        if (inliers.size() > this->bestInlierCount) {
            this->bestInlierCount = (int) inliers.size();
            this->bestHomography = H;
        }
    }

    // Generate a final Homography from all inlier matches of the best estimate
    inliers = computerInlierCount(this->bestHomography, inlierThreshold);
    vector<Point2f> img1Points;   // src
    vector<Point2f> img2Points;   // dst

    for (auto m : inliers) {
        img1Points.emplace_back(Point_<float>(m.getPoint1().getFeatureCol(), m.getPoint1().getFeatureRow()));
        img2Points.emplace_back(Point_<float>(m.getPoint2().getFeatureCol(), m.getPoint2().getFeatureRow()));
    }

    this->finalHomography = findHomography(img1Points, img2Points, 0);

    return drawMatches(inlierThreshold, this->finalHomography);
}

Point2f Stitching::project(Point2f p1, Mat H) {
    Mat point = Mat::zeros(3, 1, CV_32F);
    point.at<float>(0,0) = p1.x;
    point.at<float>(1,0) = p1.y;
    point.at<float>(2,0) = 1.0;

    Mat projection;
    H.convertTo(projection, CV_32F);

    Mat result = projection * point;

    return Point2f(result.at<float>(0,0)/result.at<float>(2,0), result.at<float>(1,0)/result.at<float>(2,0));
}

vector<Match> Stitching::computerInlierCount(Mat H, int inlierThreshold) {
    int count = 0;
    vector<Match> inliers = vector<Match>();

    for (auto match : this->matches) {
        Point2f p1 = Point_<float>(match.getPoint1().getFeatureCol(), match.getPoint1().getFeatureRow());
        Point2f p2 = Point_<float>(match.getPoint2().getFeatureCol(), match.getPoint2().getFeatureRow());
        Point2f projection = project(p1, H);

        double distance = sqrt(pow(projection.x-p2.x, 2) + pow(projection.y-p2.y, 2));

        if (distance <= inlierThreshold) {
            count++;
            inliers.emplace_back(match);
        }
    }

    return inliers;
}

Mat Stitching::drawMatches(int inlierThreshold, Mat H) {
    RNG rng(12345);
    vector<Match> inliers = computerInlierCount(H, inlierThreshold);
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

Mat Stitching::stitch() {
    Mat homInv = this->finalHomography.inv();
    Mat img1 = this->image1;
    Mat img2 = this->image2;

    img1.convertTo(img1, CV_32FC4);
    img2.convertTo(img2, CV_32FC4);

    waitKey(0);

    // Determine size of stitched image
    vector<Point2f> corners;
    int offsetx = 0;    // Create space of points to the left of img1
    int offsety = 0;    // Create space for points above img1
    int overflowx = 0;  // Create space for points to the right of img1
    int overflowy = 0;  // Create space for points below img1

    corners.emplace_back(project(Point(0, 0), homInv));
    corners.emplace_back(project(Point(img2.cols, 0), homInv));
    corners.emplace_back(project(Point(0, img2.rows), homInv));
    corners.emplace_back(project(Point(img2.cols, img2.rows), homInv));

    for (auto c : corners) {
        if (c.x < 0 && c.x < offsetx) {
            offsetx = (int) floor(c.x);
        }
        else if (c.x > img1.cols && c.x - img1.cols > overflowx) {
            overflowx = (int) ceil(c.x) - img1.cols;
        }

        if (c.y < 0 && c.y < offsety) {
            offsety = (int) floor(c.y);
        }
        else if (c.y > img1.rows && c.y - img1.rows > overflowy) {
            overflowy = (int) ceil(c.y) - img1.rows;
        }
    }

    offsetx = abs(offsetx);
    offsety = abs(offsety);

    Mat stitched = Mat::zeros(img1.rows + offsety + overflowy,
                              img1.cols + offsetx + overflowx,
                              CV_32FC4);

    // Copy img1 to stitched
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            Vec4f value = img1.at<Vec4f>(i, j);
            stitched.at<Vec4f>(i+offsety, j+offsetx) = value;
        }
    }

    for (int i = 0; i < img2.rows; i++) {
        for (int j = 0; j < img2.cols; j++) {
            Vec4f value = img2.at<Vec4f>(i, j);
            Point p = project(Point(j, i), homInv);
            stitched.at<Vec4f>(p.y+offsety, p.x+offsetx) = value;
        }
    }

    stitched.convertTo(stitched, CV_8UC4);

    return stitched;
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
