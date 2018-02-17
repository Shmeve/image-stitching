#pragma once
#define NUMBER_OF_GRIDS 16
#define NUMBER_OF_CELLS 16
#define NUMBER_OF_BINS 8
#define WINDOW_SIZE 16
#define NORMALIZING_THRESHOLD 0.2

#include <vector>

class SIFTDescriptor {
private:
    float windowX[WINDOW_SIZE][WINDOW_SIZE];
    float windowY[WINDOW_SIZE][WINDOW_SIZE];
    float bins[NUMBER_OF_GRIDS][NUMBER_OF_BINS];
public:
    SIFTDescriptor(float x[WINDOW_SIZE][WINDOW_SIZE], float y[WINDOW_SIZE][WINDOW_SIZE]);
    void generateHistograms();
    int indexForTheta(float theta);
};