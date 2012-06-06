#include "colorFeatures.h"

colorFeatures::colorFeatures() {}

colorFeatures::~colorFeatures() {}

IplImage* colorFeatures::bgr2hsv(IplImage* bgr) {
    IplImage* bgr32f, * hsv;
    
    bgr32f = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 3 );
    hsv = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 3 );
    cvConvertScale( bgr, bgr32f, 1.0 / 255.0, 0 );
    cvCvtColor( bgr32f, hsv, CV_BGR2HSV );
    cvReleaseImage( &bgr32f );
    return hsv;
}

int colorFeatures::histoBinHSV(float h, float s, float v) {
    int hd, sd, vd;
    
    /* if S or V is less than its threshold, return a "colorless" bin */
    vd = MIN( (int)(v * NV / V_MAX), NV-1 );
    if( s < S_THRESH  ||  v < V_THRESH )
        return NH * NS + vd;
    
    /* otherwise determine "colorful" bin */
    hd = MIN( (int)(h * NH / H_MAX), NH-1 );
    sd = MIN( (int)(s * NS / S_MAX), NS-1 );
    return sd * NH + hd;

}

histogram* colorFeatures::comHistogramHSV(IplImage** imgs, int n) {
    IplImage* img;
    histogram* histo;
    IplImage* h, * s, * v;
    float* hist;
    int i, r, c, bin;
    
    histo = (histogram*) malloc( sizeof(histogram) );
    
    histo->n = NH*NS + NV;
    hist = histo->histo;
    memset( hist, 0, histo->n * sizeof(float) );
    
    for( i = 0; i < n; i++ )
    {
        
        img = imgs[i];
        h = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
        s = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
        v = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
        cvCvtPixToPlane( img, h, s, v, NULL );
        
        /* increment appropriate histogram bin for each pixel */
        for( r = 0; r < img->height; r++ )
            for( c = 0; c < img->width; c++ )
            {
                bin = histoBinHSV( pixval32f( h, r, c ),
                                  pixval32f( s, r, c ),
                                  pixval32f( v, r, c ) );
                hist[bin] += 1;
            }
        cvReleaseImage( &h );
        cvReleaseImage( &s );
        cvReleaseImage( &v );
    }
    return histo;
}

void colorFeatures::normalizeHistogram(histogram* histo) {
    float* hist;
    float sum = 0, inv_sum;
    int i, n;
    
    hist = histo->histo;
    n = histo->n;
    
    /* compute sum of all bins and multiply each bin by the sum's inverse */
    for( i = 0; i < n; i++ )
        sum += hist[i];
    inv_sum = (float)1.0 / sum;
    for( i = 0; i < n; i++ )
        hist[i] *= inv_sum;
}

float colorFeatures::histoDistSq(histogram* h1, histogram* h2) {
    float* hist1, * hist2;
    float sum = 0;
    int i, n;
    
    n = h1->n;
    hist1 = h1->histo;
    hist2 = h2->histo;
    
    for( i = 0; i < n; i++ )
        sum += (float)sqrt( hist1[i]*hist2[i] );
    return (float)(1.0 - sum);

}

float colorFeatures::pixval32f(IplImage* img, int r, int c) {
    return ( (float*)(img->imageData + img->widthStep*r) )[c];
}

void colorFeatures::setpix32f(IplImage* img, int r, int c, float val) {
    ( (float*)(img->imageData + img->widthStep*r) )[c] = val;
}

float colorFeatures::likelihoodHSV( IplImage* img, int r, int c,int w, int h, histogram* ref_histo ) {
    IplImage* tmp;
    histogram* histo;
    float d_sq;
    
    /* extract region around (r,c) and compute and normalize its histogram */
    
    cvSetImageROI( img, cvRect( c - w / 2, r - h / 2, w, h ) );
    tmp = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 3 );
    cvCopy( img, tmp, NULL );
    cvResetImageROI( img );
    histo = comHistogramHSV( &tmp, 1 );
    cvReleaseImage( &tmp );
    normalizeHistogram( histo );
    
    /* compute likelihood as e^{\lambda D^2(h, h^*)} */
    d_sq = histoDistSq( histo, ref_histo );
    free(histo);
    return (float)exp( -LAMBDA * d_sq );
}