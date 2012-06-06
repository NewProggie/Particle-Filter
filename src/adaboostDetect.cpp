#include "adaboostDetect.h"
#include <stdio.h>

adaboostDetect::adaboostDetect() {
    scaleFactor = 1.1;
    minNeighbours = 3;
    flags = 0;
    minSize = cvSize(20, 20);
    maxR = 30;
    storage = cvCreateMemStorage(0);
    cascade = 0;
}

adaboostDetect::~adaboostDetect() {
    cvReleaseMemStorage(&storage);
}

int adaboostDetect::detectAndDraw(IplImage* img, CvRect** regions) {
    double t = (double) cvGetTickCount();
    int fii = 0;
    IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
    IplImage* smallImg = cvCreateImage( cvSize( cvRound (img->width/scaleFactor),
                                               cvRound (img->height/scaleFactor)), 8, 1 );
    cvCvtColor(img, gray, CV_BGR2GRAY);
    cvResize(gray, smallImg,CV_INTER_LINEAR);
    cvEqualizeHist(smallImg, smallImg);
    cvClearMemStorage(storage);
    
    int nx1, nx2, ny1, ny2;
    CvRect* nR;
    
    if (!cascade) {
        return 0;
    }
    
    CvSeq* faces = cvHaarDetectObjects( smallImg, cascade, storage, scaleFactor, minNeighbours, flags, minSize);
    for (int i=0; i<(faces ? faces->total : 0); i++) {
        if (i == 0) {
            nR = (CvRect*) malloc(1 * sizeof(CvRect));
        } else {
            nR = (CvRect*) realloc(nR, (i+1) * sizeof(CvRect));
        }
        CvRect* r = (CvRect*) cvGetSeqElem(faces, i);
        CvPoint center;
        int radius;
        center.x = cvRound((r->x + r->width * 0.5) * scaleFactor);
        center.y = cvRound((r->y + r->height * 0.5) * scaleFactor);
        radius = cvRound((r->width + r->height) * 0.25 * scaleFactor);
        nx1 = cvRound(r->x * scaleFactor);
        ny1 = cvRound(r->y * scaleFactor);
        nx2 = cvRound((r->x + r->width) * scaleFactor);
        ny2 = cvRound((r->y + r->height) * scaleFactor);
        nR[fii] = cvRect(nx1, ny1, nx2-nx1, ny2-ny1);
        CvScalar color;
        color = CV_RGB(0, 255, 0);
        cvRectangle(img, cvPoint(nx1, ny1), cvPoint(nx2, ny2), color);
        fii++;
    }
    
    *regions = nR;
    
    cvShowImage("result", img);
    cvReleaseImage(&gray);
    cvReleaseImage(&smallImg);
    t = (double) cvGetTickCount() - t;
    printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
    return fii;
}

int adaboostDetect::detectObject(IplImage* img, CvRect** regions) {
    int fii = 0;
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
    IplImage* smallImg = cvCreateImage( cvSize( cvRound (img->width/scaleFactor),
                                                cvRound (img->height/scaleFactor)), 8, 1 );
    cvCvtColor(img, gray, CV_BGR2GRAY);
    cvResize(gray, smallImg, CV_INTER_LINEAR);
    cvEqualizeHist(smallImg, smallImg);
    cvClearMemStorage(storage);
    
    int nx1, nx2, ny1, ny2;
	CvRect* nR;
    
	if(!cascade) {
        return 0;
    }
		
    CvSeq* faces = cvHaarDetectObjects( smallImg, cascade, storage, scaleFactor, minNeighbours, flags, minSize);
    for (int i=0; i<(faces ? faces->total : 0); i++) {
        CvRect* r = (CvRect*) cvGetSeqElem(faces, i);
        if (fii == 0) {
            nR = (CvRect*) malloc(1 * sizeof(CvRect));
        } else {
            nR = (CvRect*) realloc(nR, (fii+1) * sizeof(CvRect));
        }
        
        if ((r->width <= maxSize.width) && (r->height <= maxSize.height)) {
            nx1 = cvRound(r->x * scaleFactor);
            ny1 = cvRound(r->y * scaleFactor);
            nx2 = cvRound((r->x + r->width) * scaleFactor);
            ny2 = cvRound((r->y + r->height) * scaleFactor);
            nR[fii] = cvRect(nx1, ny1, nx2-nx1, ny2-ny1);
            fii++;
        }
    }
    
    *regions = nR;
    
    cvReleaseImage(&gray);
    cvReleaseImage(&smallImg);
    
    return fii;
}

int adaboostDetect::detectCheck(IplImage* img, float maxSizeDiff, float maxPosDiff, int nStages) {
    maxSizeDiff = 1.5f;
    maxPosDiff = 0.3f;
    int detCount;
    
    /* number of stages. If <= 0 all stages are used */
    int nos = -1;
    int nos0 = cascade->count;
    
    CvSeq* objects;
    cvClearMemStorage(storage);
    if (nos <= 0) {
        nos = nos0;
    }
   
    ObjectPos det;
    float distance;
    float sf = 1.1f;
    ObjectPos ref;
    cascade->count = nos;
    objects = cvHaarDetectObjects(img, cascade, storage, sf, 0);
    cascade->count = nos0;
    
    int w = img->width;
    int h = img->height;
    ref.x = 0.5f * w;
    ref.y = 0.5f * h;
    ref.width = sqrtf(0.5f * (w*w + h*h));
    ref.found = 0;
    ref.neighbours = 0;
    
    detCount = (objects ? objects->total : 0);
    int found = 0;
    for (int i=0; i<detCount; i++) {
        CvAvgComp r = *((CvAvgComp*) cvGetSeqElem(objects, i));
        det.x = 0.5f * r.rect.width + r.rect.x;
        det.y = 0.5f * r.rect.height + r.rect.y;
        det.width = sqrtf(0.5f * (r.rect.width * r.rect.width + r.rect.height * r.rect.height));
        det.neighbours = r.neighbors;
        distance = sqrtf((det.x - ref.x) * (det.x - ref.x) + (det.y - ref.y) * (det.y - ref.y));
        if ((distance < ref.width * maxPosDiff) && (det.width > ref.width / maxSizeDiff) && (det.width < ref.width * maxSizeDiff)) {
            ref.found = 1;
            ref.neighbours = MAX(ref.neighbours, det.neighbours);
            found = 1;
        }
    }
    
    return found;
}