#ifndef ADABOOST_H
#define ADABOOST_H

#include <cv.h>
#include <highgui.h>

typedef struct ObjectPos {
    float x;
    float y;
    float width;
    int found;
    int neighbours;
} ObjectPos;

/**
 Ali and M. N. Dailey, "Multiple human tracking in high-density crowds"
 Abschnitt 2.2 Detection */
class adaboostDetect {
public:
    adaboostDetect();
    ~adaboostDetect();
    double scaleFactor;
    int minNeighbours;
    int flags; /* CV_HAAR_DO_CANNY_PRUNING */
    CvSize minSize;
    CvSize maxSize;
    int maxR;
    int findBiggestObject;
    CvMemStorage* storage;
    CvHaarClassifierCascade* cascade;
    int detectAndDraw(IplImage* img, CvRect** regions);
    int detectObject(IplImage* img, CvRect** regions);
    int detectCheck(IplImage* img, float maxSizeDiff, float maxPosDiff, int nStages);
};

#endif