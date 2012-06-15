#include <cv.h>
#include <highgui.h>
#include "constants.h"
#include <iostream>

#include "adaboostDetect.h"
#include "particleFilter.h"
#include "tracker.h"
#include "colorFeatures.h"

using namespace std;

int initDetection(adaboostDetect* detect) {
    const char* cascadeName = "cascades.xml";
    detect->cascade = cvLoadHaarClassifierCascade(cascadeName, cvSize(20, 20));
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
    const char* vidName = "Fussgaengerzone.m4v";
    IplImage* frame;
    int frameNo = 0, nHeads = 0;
    CvRect* regions;
    adaboostDetect* detect = new adaboostDetect;
    tracker* hTrack = new tracker;
    colorFeatures cf;
    CvCapture* capture = cvCaptureFromAVI(vidName);

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
            hTrack->next(frame, detect, vidName);
            regions = 0;
            int n = detect->detectObject(frame, &regions);
            nHeads += n;
            hTrack->addObjects(frame, regions, n);
        }
        
        hTrack->showResults(frame, 0);
        cvShowImage(WINDOW_TITLE, frame);
        
//        if ((cvWaitKey(10) & 255) == 27) {
//            break;
//        }
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