#include "SIFTDescriptor.h"
#include <cmath>

/**
 * Constructor
 *
 * @param x window of x derivative values
 * @param y window of y derivative values
 */
SIFTDescriptor::SIFTDescriptor(float x[WINDOW_SIZE][WINDOW_SIZE], float y[WINDOW_SIZE][WINDOW_SIZE], int row, int col) {
    // Init neighbourhood window
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = 0; j < WINDOW_SIZE; j++) {
            this->windowX[i][j] = x[i][j];
            this->windowY[i][j] = y[i][j];
        }
    }

    // Init bins to 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = 0; j < NUMBER_OF_BINS; j++) {
            bins[i][j] = 0.0;
        }
    }

    // Track interest points origin
    this->featureRow = row;
    this->featureCol = col;
}

/**
 * Generate the histograms for all grids within the window
 *
 * @return void
 */
void SIFTDescriptor::generateHistograms() {
    float m;
    int a;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int gX = i/4;               // X Grid Value

        for (int j = 0; j < WINDOW_SIZE; j++) {
            int gY = j/4;           // Y Grid Value
            int gIndex = gX*4+gY;   // Map 2D position to 1D bins array
            int bin;                // Bin index
            float x = windowX[i][j];
            float y = windowY[i][j];

            m = sqrt((x*x)+(y*y));
            a = (int) ((atan(y/x)*180)/M_PI) % 360;
            bin = indexForTheta(a);

            bins[gIndex][bin] += (m > 0.2) ? 0.2 : m;
        }
    }
}

/**
 * Maps theta to a bin index
 *
 * @param theta calculated angle from -pi/2 to pi/2
 * @return int
 */
int SIFTDescriptor::indexForTheta(float theta) {
    if (theta < 0) {
        theta = 360 + theta;
    }

    if (theta < 45) {
        return 0;
    } else if (theta < 90) {
        return 1;
    } else if (theta < 135) {
        return 2;
    } else if (theta < 180) {
        return 3;
    } else if (theta < 225) {
        return 4;
    } else if (theta < 270) {
        return 5;
    } else if (theta < 315) {
        return 6;
    } else {
        return 7;
    }
}

/**
 * Return a distance (SSD) between this descriptor and its input
 *
 * @param f1 Feature Descriptor to compare feature
 * @return double
 */
double SIFTDescriptor::SSD(SIFTDescriptor f1) {
    double score = 0;

    for (int i = 0; i < NUMBER_OF_GRIDS; i++) {
        for (int j = 0; j < NUMBER_OF_BINS; j++) {
            score += pow((f1.bins[i][j] - bins[i][j]), 2);
        }
    }

    return score;
}

int SIFTDescriptor::getFeatureRow() const { return featureRow; }
int SIFTDescriptor::getFeatureCol() const { return featureCol; }
