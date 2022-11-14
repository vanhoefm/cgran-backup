#ifndef FLAT_RAYLEIGH
#define FLAT_RAYLEIGH

#include "Random.h"
/*****************************************************************************
 * Author: Christos Komninakis                                               *
 * Date: January 4, 2000.                                                    *
 * Content: header file for the CLASS "flat_rayleigh".                       *
 *****************************************************************************/
#include <iostream>
#include <fstream>
#include "Complex.h"

using namespace std;

class flat_rayleigh
{
  protected:
  long chan_seed;	/* the random channel seed */
  bool IndepFlag;	/* 1 if i.i.d. blocks, 0 if continuous across blocks */
  float PWR;		/* the power of the fading waveform */
  Complex chan_val;	/* the flat fading complex coefficient of the channel */
  int K;        /* the number of biquads for my ellipsoid-algorithm design */
  int H, H2;    /* number of interpolating coefs (one sided). H2=2*H */
  int I;        /* the interpolation factor, given by 0.2/fD */
  int last_i, IP;   /* helpful indices: IP=Insertion Point into buff_f */
  float *a, *b, *c, *d, Ao;  /* biquad coefs, and gain */
  float **sinc_matrix;
  Complex **st;    /* state of the K biquads */
  Complex *buff_f; /* the rarely sampled output of the 0.2-Doppler prototype */
  float *buff_sinc; /* pointer to the pertinent row of the interp. matrix */
  
  public:
  flat_rayleigh(int seeed, float fD, float pwr, bool flag_indep);
  ~flat_rayleigh()
    {
      delete [] a;
      delete [] b;
      delete [] c;
      delete [] d;
      delete [] st;
      delete [] buff_f;
      delete [] sinc_matrix;
    }
  void pass_through(int length, Complex *inp, Complex *outp);
  void pass_through(int length, Complex *inp, Complex *outp, Complex *csi);
  void pass_through(int length, Complex *inp, Complex *outp, float *amp_csi);
};

#endif
