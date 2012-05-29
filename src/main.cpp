#include "cv.h"
#include "highgui.h"
#include <iostream>

using namespace std;

int main() {
    
    char* WINDOW_TITLE = "Particle Filter";
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
    assert(capture);
    
    cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    while (1) {
        IplImage* frame = cvQueryFrame(capture);
        if (!frame) {
            cerr << "ERROR: frame is null" << endl;
            break;
        } else {
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