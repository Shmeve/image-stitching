#include "SIFTDescriptor.h"
#include <cmath>

/**
 * Constructor
 *
 * @param x window of x derivative values
 * @param y window of y derivative values
 */
SIFTDescriptor::SIFTDescriptor(float x[WINDOW_SIZE][WINDOW_SIZE], float y[WINDOW_SIZE][WINDOW_SIZE]) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = 0; j < WINDOW_SIZE; j++) {
            this->windowX[i][j] = x[i][j];
            this->windowY[i][j] = y[i][j];
        }
    }
}

/**
 * Generate the histograms for all grids within the window
 *
 * @return void
 */
void SIFTDescriptor::generateHistograms() {
    float m, a;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int gX = i/4;               // X Grid Value

        for (int j = 0; j < WINDOW_SIZE; j++) {
            int gY = j/4;           // Y Grid Value
            int gIndex = (gX/4)+gY; // Map 2D position to 1D bins array
            int bin;                // Bin index
            float x = windowX[i][j];
            float y = windowY[i][j];

            m = sqrt((x*x)+(y*y));
            a = atan(x/y);
            bin = indexForTheta(a);

            bins[gIndex][bin] += m;
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
    if (theta < -M_PI/4) {
        return 0;
    } else if (theta < -M_PI/6) {
        return 1;
    } else if (theta < -M_PI/8) {
        return 2;
    } else if (theta < 0) {
        return 3;
    } else if (theta < M_PI/8) {
        return 4;
    } else if (theta < M_PI/6) {
        return 5;
    } else if (theta < M_PI/4) {
        return 6;
    } else {
        return 7;
    }
}
