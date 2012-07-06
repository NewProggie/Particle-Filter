#ifndef COLORFEATURES_H
#define COLORFEATURES_H

#include <cv.h>

/** number of bins of HSV in histogram */
#define NH 10
#define NS 10
#define NV 10

/** max HSV values */
#define H_MAX 360.0
#define S_MAX 1.0
#define V_MAX 1.0

/** low thresholds on saturation and value for histogramming */
#define S_THRESH 0.1
#define V_THRESH 0.2

/* distribution parameter */
#define LAMBDA 20

typedef struct {
    float histo[NH*NS + NV];   /** histogram array */
    int n;                     /** length of histogram array */
} histogram;

/** Ali and M. N. Dailey, "Multiple human tracking in high-density crowds"
 Abschnitt 2.5 Appearance Model */
class colorFeatures {
public:
    colorFeatures();
    ~colorFeatures();
    /** Change image to hsv color model */
    IplImage* bgr2hsv(IplImage* bgr);
    /** compute hsv histogram from image */
    histogram* comHistogramHSV(IplImage* img, int n);
    /** normalizes histogram */
    void normalizeHistogram(histogram* histo);
    /** return the square distance of given histograms */
    float histoDistSq(histogram* h1, histogram* h2);
    /** return pixel value of given location */
    float pixval32f(IplImage* img, int r, int c);
    /** set pixel value of given location */
    void setpix32f(IplImage* img, int r, int c, float val);
    float likelihoodHSV( IplImage* img, int r, int c,int w, int h, histogram* ref_histo );
private:
    int histoBinHSV(float h, float s, float v);
};


#endif