#pragma once

#include "../FeatureMatching/SIFT/SIFTDescriptor.h"

class Match {
private:
    SIFTDescriptor point1;
    SIFTDescriptor point2;
public:
    Match(const SIFTDescriptor &point1, const SIFTDescriptor &point2);

    const SIFTDescriptor &getPoint1() const;

    void setPoint1(const SIFTDescriptor &point1);

    const SIFTDescriptor &getPoint2() const;

    void setPoint2(const SIFTDescriptor &point2);
};