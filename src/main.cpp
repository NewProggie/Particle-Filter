#include <cv.h>
#include <highgui.h>
#include <iostream>

#include "adaboostDetect.h"
#include "particleFilter.h"
#include "tracker.h"
#include "constants.h"
#include "colorFeatures.h"

using namespace std;

int initDetection(adaboostDetect* detect) {
    detect->cascade = cvLoadHaarClassifierCascade(CASCADE_XML_PATH, cvSize(20, 20));
    if (!detect->cascade) {
        return 0;
    }
    detect->flags = CV_HAAR_DO_CANNY_PRUNING;
    detect->maxR = 23;
    detect->minNeighbours = 3;
    detect->minSize = cvSize(10, 10);
    detect->maxSize = cvSize(40, 40);
    detect->scaleFactor = 1.1;
    
    return 1;
}

int main() {
    /** Current Frame as IplImage */
    IplImage* frame;
    /** current frame number */
    int frameNo = 0;
    /** number of detected heads */
    int nHeads = 0;
    /** number of current detected heads */
    int n;
    /** list of regions around each detected head */
    CvRect* regions;
    adaboostDetect* detect = new adaboostDetect;
    tracker* hTrack = new tracker;
    colorFeatures cf;
    CvCapture* capture = cvCaptureFromAVI(VIDEO_PATH);

    assert(capture);
    assert(initDetection(detect));
    
    cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    while (1) {
        frame = cvQueryFrame(capture);
        assert(frame);
        if (frameNo == 0) {
            nHeads = detect->detectObject(frame, &regions);
            hTrack->initTracker(frame, regions, nHeads, 20);
        } else {
            hTrack->next(frame);
            regions = 0;
            n = detect->detectObject(frame, &regions);
            nHeads += n;
            hTrack->addObjects(frame, regions, n);
        }
        
        hTrack->showResults(frame);
        cvShowImage(WINDOW_TITLE, frame);
        frameNo++;
        
        if ((cvWaitKey(10) & 255) == 27) {
            break;
        }
    }
    
    if (capture) {
        cvReleaseCapture(&capture);
    }
    
    hTrack->~tracker();
    free(hTrack);
    free(detect);
    
    cvReleaseCapture(&capture);
    cvDestroyWindow(WINDOW_TITLE);
    
    return 0;
}