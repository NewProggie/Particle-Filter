#include <cv.h>
#include <highgui.h>
#include "constants.h"
#include <iostream>

#include "colorFeatures.h"

using namespace std;

int main() {
    
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
    assert(capture);
    colorFeatures cf;
    
    cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    while (1) {
        IplImage* frame = cvQueryFrame(capture);
        if (!frame) {
            cerr << "ERROR: frame is null" << endl;
            break;
        } else {
            frame = cf.bgr2hsv(frame);
            cvShowImage(WINDOW_TITLE, frame);
        }
        
        if ((cvWaitKey(10) & 255) == 27) {
            break;
        }
    }
    
    cvReleaseCapture(&capture);
    cvDestroyWindow(WINDOW_TITLE);
    
    return 0;
}