#pragma once

#include <vector>

class SIFTDescriptor {
private:
    const static int WINDOW_SIZE = 16;
    const static int NUMBER_OF_GRIDS = 16;
    const static int NUMBER_OF_BINS = 8;
    float windowX[WINDOW_SIZE][WINDOW_SIZE];
    float windowY[WINDOW_SIZE][WINDOW_SIZE];
    float bins[NUMBER_OF_GRIDS][NUMBER_OF_BINS];
public:
    SIFTDescriptor(float x[WINDOW_SIZE][WINDOW_SIZE], float y[WINDOW_SIZE][WINDOW_SIZE]);
    void generateHistograms();
    int indexForTheta(float theta);
};