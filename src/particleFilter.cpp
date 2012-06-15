#include "particleFilter.h"

particleFilter::particleFilter() {
    nParticles = 0;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
}

particleFilter::~particleFilter() {
    
}

void particleFilter::resetCCWeights() {
    for (int i=0; i < nParticles; i++) {
        particles[i].w = 1-particles[i].w;
        particles[i].w = (float) exp(-3 * particles[i].w);
    }
}

CvRect particleFilter::getParticleRect() {
    CvRect rect;
    rect.x = cvRound(particles[0].x - 0.5 * particles[0].s * particles[0].width);
    rect.y = cvRound(particles[0].y - 0.5 * particles[0].s * particles[0].height);
    rect.width = cvRound(particles[0].s * particles[0].width);
    rect.height = cvRound(particles[0].s * particles[0].height);
    
    return rect;
}

CvPoint particleFilter::getParticleCenter() {
    return cvPoint(cvRound(particles[0].x), cvRound(particles[0].y));
}

void particleFilter::initParticles(CvRect region, int particlesPerObject) {
    nParticles = particlesPerObject;
    int width = region.width;
    int height = region.height;
    int x = (float) region.x + width / 2;
    int y = (float) region.y + height / 2;
    for (int i=0; i < nParticles; i++) {
        particles[i].x0 = particles[i].xp = particles[i].x = x;
        particles[i].y0 = particles[i].yp = particles[i].y = y;
        particles[i].sp = particles[i].s = 1.0;
        particles[i].width = width;
        particles[i].height = height;
        particles[i].w = 0;
    }
}

void particleFilter::resetParticles(CvRect region) {
    int width = region.width;
    int height = region.height;
    int x = (float) region.x + width / 2;
    int y = (float) region.y + height / 2;
    for (int i=0; i < nParticles; i++) {
        particles[i].x = x;
        particles[i].y = y;
        particles[i].s = 1.0;
        particles[i].width = width;
        particles[i].height = height;
        particles[i].w = 1/20;
    }
}

void particleFilter::transition(int w, int h) {
    for (int i=0; i < nParticles; i++) {
        particles[i] = calTransition(particles[i], w, h, rng);
    }
}

void particleFilter::normalizeWeights() {
    float sum = 0;
    
    for (int i=0; i < nParticles; i++) {
        sum += particles[i].w;
    }
    
    for (int i=0; i < nParticles; i++) {
        particles[i].w /= sum;
    }
}

int particleCmp(const void* p1, const void* p2) {
    particle* _p1 = (particle*)p1;
    particle* _p2 = (particle*)p2;
    
    if( _p1->w > _p2->w )
        return -1;
    if( _p1->w < _p2->w )
        return 1;
    return 0;
}

void particleFilter::resample() {
    int np, k = 0;
    particle * newParticles;
    qsort(particles, nParticles, sizeof(particle), &particleCmp);
    
    newParticles = (particle*) malloc(nParticles * sizeof(particle));
    for (int i=0; i < nParticles; i++) {
        np = cvRound(particles[i].w * nParticles);
        for (int j=0; j < np; j++) {
            newParticles[k++] = particles[i];
            if (k == nParticles) {
                goto exit;
            }
        }
    }
    
    while (k < nParticles) {
        newParticles[k++] = particles[0];
    }
    
    exit:
    for (int i=0; i < nParticles; i++) {
        particles[i] = newParticles[i];
    }
    free(newParticles);
}

void particleFilter::displayParticles(IplImage* img, CvScalar nColor, CvScalar hColor, int param) {
    CvScalar color;
    if (param == SHOW_ALL) {
        for (int i=nParticles-1; i >= 0; i--) {
            if (i == 0) {
                color = hColor;
            } else {
                color = nColor;
            }
            displayParticle(img, particles[i], color);
        }
    } else if (param == SHOW_SELECTED) {
        color = hColor;
        displayParticle(img, particles[0], color);
    }
}

void particleFilter::updateWeight(IplImage* frameHSV, histogram* objectHisto) {
    colorFeatures cf;
    float s;
    for (int i=0; i < nParticles; i++) {
        s = particles[i].s;
        particles[i].w = cf.likelihoodHSV(frameHSV, cvRound(particles[i].y), cvRound(particles[i].x), cvRound(particles[i].width * s), cvRound(particles[i].height * s), objectHisto); 
    }
}

particle particleFilter::calTransition(particle p, int w, int h, gsl_rng* rng) {
    particle pn;
    /** double x = A1 * (p.x - p.x0) + A2 * (p.xp - p.x0) + B0 * gsl_ran_gaussian(rng, TRANS_X_STD) + p.x0;
    double y = A1 * (p.y - p.y0) + A2 * (p.yp - p.y0) + B0 * gsl_ran_gaussian(rng, TRANS_Y_STD) + p.y0;
    double s = A1 * (p.s - 1.0) + A2 * (p.sp - 1.0) + B0 * gsl_ran_gaussian(rng, TRANS_S_STD) + 1.0; */
    
    pn.s = 1.0;
    pn.xp = p.x;
    pn.yp = p.y;
    pn.sp = p.s;
    pn.x0 = p.x0;
    pn.y0 = p.y0;
    pn.width = p.width;
    pn.height = p.height;
    pn.w = 0;
    
    return pn;
}

void particleFilter::displayParticle(IplImage* img, particle p, CvScalar color) {
    int x0 = cvRound( p.x - 0.5 * p.s * p.width );
    int y0 = cvRound( p.y - 0.5 * p.s * p.height );
    int x1 = x0 + cvRound( p.s * p.width );
    int y1 = y0 + cvRound( p.s * p.height );
    
    cvRectangle( img, cvPoint( x0, y0 ), cvPoint( x1, y1 ), color, 2, 8, 0 );

}
