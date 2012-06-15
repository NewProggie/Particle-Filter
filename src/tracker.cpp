#include "tracker.h"

tracker::tracker() {
    nTrajectories = 0;
    frameNo = 0;
}

tracker::~tracker() {
    delete [] trajectories;
}

void tracker::initTracker(IplImage* frame, CvRect* regions, int nRegions, int particlesPerObject) {
    colorFeatures cf;
    p_perObject = particlesPerObject;
    nTrajectories = nRegions;
    IplImage* frameHSV = cf.bgr2hsv(frame);
    
    trajectories = (trajectory*) malloc(nRegions * sizeof(trajectory));
    for (int i=0; i < nTrajectories; i++) {
        particleFilter* pf = new particleFilter();
        pf->initParticles(regions[i], particlesPerObject);
        trajectories[i].object = pf;
        trajectories[i].object->objectID = 1;
        trajectories[i].object->weight = 1;
        trajectories[i].stratFrame = 0;
        trajectories[i].histo = computeHistogram(frameHSV, regions[i]);
        trajectories[i].points = (CvPoint*) malloc(sizeof(CvPoint));
        trajectories[i].points[frameNo] = trajectories[i].object->getParticleCenter();
    }
    
    frameNo++;
}

void tracker::mergeTrack() {
    float d = 0;
    float x, y, x1, y1;
    for (int i=0; i < nTrajectories; i++) {
        x = trajectories[i].object->particles[0].x;
        y= trajectories[i].object->particles[0].y;
        for (int j=0; j < nTrajectories; j++) {
            x1 = trajectories[j].object->particles[0].x;
            y1 = trajectories[j].object->particles[0].y;
            
            d = sqrtf((x1-x) * (x1-x) + (y1-y) * (y1-y));
            if (d < 20) {
                if (trajectories[i].stratFrame <= trajectories[j].stratFrame) {
                    trajectories[i].object->weight = 1;
                    trajectories[j].object->weight -= 1;
                } else {
                    trajectories[j].object->weight = 1;
                    trajectories[i].object->weight -= 1;
                }
            }
        }
    }
}

void tracker::removeTrack(int n) {
    for (int i=n; i < nTrajectories; i++) {
        trajectories[i] = trajectories[i+1];
    }
    nTrajectories--;
}

void tracker::next(IplImage* frame, adaboostDetect* adaboost, const char* filename) {
    colorFeatures cf;
    IplImage* frameHSV = cf.bgr2hsv(frame);
    int w = frame->width;
    int h = frame->height;
    for (int i=0; i < nTrajectories; i++) {
        trajectories[i].object->transition(w, h);
        trajectories[i].object->updateWeight(frameHSV, trajectories[i].histo);
        trajectories[i].object->normalizeWeights();
        trajectories[i].object->resample();
        trajectories[i].points = (CvPoint*) realloc(trajectories[i].points, (frameNo+1) * sizeof(CvPoint));
        trajectories[i].points[frameNo] = trajectories[i].object->getParticleCenter();
    }
    
    mergeTrack();
    cvReleaseImage(&frameHSV);
    frameNo++;
}

void tracker::updateObjectWeights(IplImage* frame, adaboostDetect* adaboost) {
    IplImage* temp = 0;
    colorFeatures cf;
    for (int i=0; i < nTrajectories; i++) {
        int w = cvRound(trajectories[i].object->particles[0].width);
        int h = cvRound(trajectories[i].object->particles[0].height);
        int x = cvRound(trajectories[i].object->particles[0].x) - w/2;
        int y = cvRound(trajectories[i].object->particles[0].y) - h/2;
        
        cvSetImageROI(frame, cvRect(x-5, y-5, w+10, h+10));
        temp = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, frame->nChannels);
        cvCopy(frame, temp, NULL);
        CvRect* rgs;
        if (adaboost->detectObject(temp, &rgs)) {
            if (trajectories[i].object->weight < 1) {
                trajectories[i].object->weight += 0.5;
            }
        } else {
            trajectories[i].object->weight -= 0.5;
            if (trajectories[i].object->weight < -4) {
                removeTrack(i);
            }
        }
        
        cvReleaseImage(&temp);
        cvResetImageROI(frame);
    }
}

void tracker::addObjects(IplImage* frame, CvRect* regions, int nRegions) {
    if (nRegions < 1) {
        return;
    }
    
    colorFeatures cf;
    IplImage* frameHSV = cf.bgr2hsv(frame);
    int dm = 0, tr = 0, rs = 0, nw = 0;
    int trFound[200];
    std::fill_n(trFound, nTrajectories, 0);
    for (int i = nTrajectories; i < nTrajectories + nRegions; i++) {
        int x = (float) regions[i-nTrajectories].x + regions[i-nTrajectories].width/2;
        int y = (float) regions[i-nTrajectories].y + regions[i-nTrajectories].height/2;
        for (int j = 0; j < nTrajectories; j++) {
            int x1 = trajectories[j].object->particles[0].x;
            int y1 = trajectories[j].object->particles[0].y;
            int d = sqrtf((x1-x) * (x1-x) + (y1-y) * (y1-y));
            if (j == 0) {
                dm = d;
            }
            if (d < dm) {
                tr = j;
                dm = d;
            }
        }
        
        if (dm < 20) {
            trajectories[tr].object->resetParticles(regions[i-nTrajectories]);
            frameHSV = cf.bgr2hsv(frame);
            trajectories[tr].histo = computeHistogram(frameHSV, regions[i-nTrajectories]);
            trajectories[tr].object->weight = 1;
            trFound[tr] = 1;
            rs++;
        } else {
            trajectories = (trajectory*) realloc (trajectories, (nTrajectories+nw+1) * sizeof(trajectory));
            particleFilter* pf = new particleFilter;
            pf->initParticles(regions[i-nTrajectories], p_perObject);
            trajectories[nw+nTrajectories].stratFrame = frameNo;
            trajectories[nw+nTrajectories].object = pf;
            trajectories[nw+nTrajectories].object->objectID = nw+nTrajectories;
            trajectories[nw+nTrajectories].object->weight = 0.5;
            trajectories[nw+nTrajectories].histo = computeHistogram(frameHSV, regions[i-nTrajectories]);
            trajectories[nw+nTrajectories].points = (CvPoint*) malloc(frameNo * sizeof(CvPoint));
            trajectories[nw+nTrajectories].points[frameNo] = trajectories[nw+nTrajectories].object->getParticleCenter();
            nw++;
        }
    }
    
    for (int j = 0; j < nTrajectories; j++) {
        if (trFound[j] == 1) {
            if (trajectories[j].object->weight < 1) {
                trajectories[j].object->weight += 0.5;
            }
        } else {
            trajectories[j].object->weight -= 0.5;
        }
    }
    
    nTrajectories = nTrajectories + nw;
    for (int i = 0; i < nTrajectories; i++) {
        if (trajectories[i].object->weight < -3) {
            removeTrack(i);
        }
    }
}

void tracker::showResults(IplImage* frame, int param) {
    for (int i=0; i < nTrajectories; i++) {
        CvScalar color = CV_RGB(255, 0, 0);
        if (trajectories[i].object->weight > 0.5) {
            trajectories[i].object->displayParticles(frame, CV_RGB(0, 0, 255), color, SHOW_SELECTED);
            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_PLAIN|CV_FONT_ITALIC, 1, 1, 0, 1);
            char buffer [4];
            sprintf (buffer, "%d",trajectories[i].object->objectID );
            cvPutText(frame,buffer, cvPoint( cvRound(trajectories[i].object->particles[0].x)+5, cvRound(trajectories[i].object->particles[0].y)+5 ), &font,	cvScalar(255,255,255));
        }
    }
}

IplImage* tracker::subtractObjects(IplImage* frame) {
    IplImage* tmp = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
    cvCopy(frame, tmp);
    int w = frame->width;
    int h = frame->height;
    for (int i=0; i < nTrajectories; i++) {
        if (trajectories[i].object->weight < 0.5) {
            continue;
        }
        CvRect r = trajectories[i].object->getParticleRect();
        for (int k=r.y; k<r.y+r.height;k++) {
            for (int j=r.x; j<r.x+r.width;j++) {
                if ((k>=0)&&(k<h)&&(j>=0)&&(j<w)) {
                    ((uchar *)(tmp->imageData + k*tmp->widthStep))[j*tmp->nChannels + 0] = 0;
                    ((uchar *)(tmp->imageData + k*tmp->widthStep))[j*tmp->nChannels + 1] = 0;
                    ((uchar *)(tmp->imageData + k*tmp->widthStep))[j*tmp->nChannels + 2] = 0;
                }
            }
        }
    }
    return tmp;
}

histogram* tracker::computeHistogram(IplImage* frame, CvRect region) {
    colorFeatures cf;
    cvSetImageROI(frame, region);
    IplImage* tmp = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, 3);
    cvCopy(frame, tmp, NULL);
    cvResetImageROI(frame);
    histogram* hist = cf.comHistogramHSV(&tmp, 1);
    cf.normalizeHistogram(hist);
    cvReleaseImage(&tmp);
    
    return hist;
}
