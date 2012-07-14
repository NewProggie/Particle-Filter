#ifndef TRACKER_H
#define TRACKER_H

#include "cv.h"
#include "particleFilter.h"
#include "adaboostDetect.h"

typedef struct trajectory {
    CvPoint* points;
    particleFilter* object;
    histogram* histo;
    CvMat* CCV;
    int startFrame;
} trajectory;

/** Ali and M. N. Dailey, "Multiple human tracking in high-density crowds"
    Abschnitt 2.4 Motion Model */
class tracker {
public:
    tracker();
    ~tracker();
    /**
     * Initializes trajectory for each detected head
     * @param frame current video frame
     * @param regions region for each detected head
     * @param nRegions number of detected heads
     * @param particlesPerObject number of particles for each detected object
     */
    void initTracker(IplImage* frame, CvRect* regions, int nRegions, int particlesPerObject);
    /**
     * Merge trajectories
     */
    void mergeTrack();
    /**
     * remove trajectory
     */
    void removeTrack(int n);
    /**
     * Converts current frame to hsv and updates particle filter
     * @param frame current video frame
     */
    void next(IplImage* frame);
    /**
     * Updates object weights and removes trajectories whose occlusion count is
     * above threshold
     * @param frame current video frame
     * @param adaboost current instance of adaboost detection
     */
    void updateObjectWeights(IplImage* frame, adaboostDetect* adaboost);
    /**
     * Adds new found objects to the set of tracked persons and updates
     * occlusion counts for trajectories and removes trajectories whose
     * occlusion count is above threshold
     * @param frame current video frame
     * @param regions region for each detected head
     * @param nRegions number of detected heads
     */
    void addObjects(IplImage* frame, CvRect* regions, int nRegions);
    /**
     * Paints a rectangle around each detected head together with its specific ID
     * @param frame current video frame
     */
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
    /**
     * Calculates histogram for given frame and region
     * @param frame current video frame
     * @param region region for which histogram will be calculated
     */
    histogram* computeHistogram(IplImage* frame, CvRect region);
};

#endif