#include "Match.h"

Match::Match(const SIFTDescriptor &point1, const SIFTDescriptor &point2) : point1(point1), point2(point2) {
    this->point1 = point1;
    this->point2 = point2;
}

const SIFTDescriptor &Match::getPoint1() const {
    return point1;
}

void Match::setPoint1(const SIFTDescriptor &point1) {
    Match::point1 = point1;
}

const SIFTDescriptor &Match::getPoint2() const {
    return point2;
}

void Match::setPoint2(const SIFTDescriptor &point2) {
    Match::point2 = point2;
}
