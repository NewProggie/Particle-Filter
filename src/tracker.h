#ifndef TRACKER_H
#define TRACKER_H

#include "cv.h"
#include "particleFilter.h"
#include "adaboostDetect.h"

/* Ali and M. N. Dailey, "Multiple human tracking in high-density crowds"
 Abschnitt 2.4 Motion Model */

typedef struct trajectory {
    CvPoint* points;
    particleFilter* object;
    histogram* histo;
    CvMat* CCV;
    int startFrame;
} trajectory;

class tracker {
public:
    tracker();
    ~tracker();
    void initTracker(IplImage* frame, CvRect* regions, int nRegions, int particlesPerObject);
    void mergeTrack();
    void removeTrack(int n);
    void next(IplImage* frame);
    void updateObjectWeights(IplImage* frame, adaboostDetect* adaboost);
    void addObjects(IplImage* frame, CvRect* regions, int nRegions);
    void showResults(IplImage* frame);
    IplImage* subtractObjects(IplImage* frame);
    
    
    trajectory* trajectories;
    int nTrajectories;
    int p_perObject;
    int frameNumber;
    int nbins;
    int ccvth;
    int mode;
private:
    histogram* computeHistogram(IplImage* frame, CvRect region);
};

#endif