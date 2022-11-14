/* -*- c++ -*- */
/*
 * Copyright 2004,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dabp_ofdm_demod.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>
#include <cstring>

dabp_ofdm_demod_sptr dabp_make_ofdm_demod(int mode)
{
    return dabp_ofdm_demod_sptr(new dabp_ofdm_demod(mode));
}

dabp_ofdm_demod::dabp_ofdm_demod(int mode)
    : gr_block("ofdm_demod", 
		gr_make_io_signature2(2, 2, sizeof(gr_complex), sizeof(char)), 
        gr_make_io_signature2(2, 2, sizeof(float), sizeof(char))), 
    d_param(mode),
    d_L(d_param.get_L()), d_f(d_param.get_f()), d_fa(d_param.get_fa()), 
    d_Tu(d_param.get_Tu()), d_delta(d_param.get_delta()), d_K(d_param.get_K()), 
    d_Tnull(d_param.get_Tnull()), d_Nfft(d_param.get_Nfft()), 
    d_Ts(d_param.get_Ts()), d_Ndel(d_param.get_Ndel()), 
    d_Nnull(d_param.get_Nnull()), d_Nfrm(d_param.get_Nfrm()),
    d_Nficsyms(d_param.get_Nficsyms()),
    d_freqint(mode,d_Nfft)
{
    assert(mode>=1 && mode<=4);
    
	set_relative_rate(2.0*d_K*(d_L-1)/d_Nfrm);
    set_output_multiple(2*d_K*(d_L-1)); // output one frame at a time
	
	d_fftin = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*d_Nfft);
	d_fftout = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*d_Nfft);
	d_plan = fftwf_plan_dft_1d(d_Nfft,d_fftin,d_fftout,FFTW_FORWARD, FFTW_ESTIMATE);
    d_samples_to_consume_per_frame=(d_Ts*d_L+d_Tnull/2)/d_f*d_fa;
}

dabp_ofdm_demod::~dabp_ofdm_demod()
{
	fftwf_destroy_plan(d_plan);
	fftwf_free(d_fftin);
	fftwf_free(d_fftout);
}

void dabp_ofdm_demod::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%(2*d_K*(d_L-1))==0);
	int nblocks=noutput_items/(2*d_K*(d_L-1));
    int input_required=nblocks*d_Nfrm;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_ofdm_demod::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%(2*d_K*(d_L-1))==0);
    int nblocks=noutput_items/(2*d_K*(d_L-1));
    const gr_complex *in0=(const gr_complex *)input_items[0];
	const char * in1=(const char *)input_items[1];
	float *out0=(float*)output_items[0];
    char *out1=(char *)output_items[1];
    
    int i,j,k;
    int gidx, midx;
    int nconsumed=0, nproduced=0; // the number of consumed input items & produced output items
    
    while(noutput_items-nproduced>=(2*d_K*(d_L-1)) && ninput_items[0]-nconsumed>=d_Nfrm) { // loop if what's left is at least one dab frame
		while(!*in1 && ninput_items[0]-nconsumed>=d_samples_to_consume_per_frame) {// discard samples before PRS
			nconsumed++;
			in1++;
			in0++;
		}
		if(!*in1) { // no more frame
			break;
		}
		nconsumed++;
		in1++;
		in0++;
		// now in0/1 point to the first sample of PRS
		// start to process one dab frame. This will consume (d_Ts*d_L+d_Tnull/2)/d_f*d_fa samples
		// fine CFO
		gr_complex gamma(0.0);
		for(i=0;i<d_L;i++) {
			gidx=(int)(d_Ts/d_f*d_fa*(i+1))-1; // index to the last sample of each ofdm symbol
			for(j=0;j<d_Ndel-1;j++) {
				gamma+=std::conj(in0[gidx-d_Ndel+2+j])*in0[gidx-d_Ndel+2+j-d_Nfft];
			}
		}
		float cfo=std::arg(gamma);
		gr_complex x[d_Nfft], z[d_Nfft], y[d_Nfft]; // one OFDM symbol with CP removed
		int ctr=0;
		float maxpz=-1e20;
		for(i=0;i<d_L;i++) {
			midx=(d_Nfft+d_Ndel)*(i+1)-1; // index to the last sample of each ofdm symbol
			// correcting CFO
			for(j=0;j<d_Nfft;j++) {
                x[j]=in0[midx-d_Nfft-TSYNC_TOL+j+1]*std::polar(1.0f,cfo*(midx-d_Nfft-TSYNC_TOL+j+1.0f)/d_Nfft);
            }
			
			// convert to frequency domain
			memcpy(d_fftin,x,sizeof(gr_complex)*d_Nfft);
			fftwf_execute(d_plan);
			memcpy(x,d_fftout,sizeof(gr_complex)*d_Nfft);
            
			// coarse CFO
			if(i==0) { // only base on PRS
				for(j=-MAXSC;j<=MAXSC;j++) {
					float s=0;
					for(k=1;k<=d_K/2;k++) {
						s+=std::norm(x[(j+k>=0)? (j+k)%d_Nfft : (j+k)%d_Nfft+d_Nfft])
							+std::norm(x[(j-k>=0)? (j-k)%d_Nfft : (j-k)%d_Nfft+d_Nfft]);
					}
					if(s>maxpz) {
						maxpz=s;
						ctr=j;
					}
				}
			}else { // non PRS
				// differential demod
				for(j=0;j<d_Nfft;j++) {
					k=(ctr+j>=0)? (ctr+j)%d_Nfft : (ctr+j)%d_Nfft+d_Nfft;
					y[j]=x[k]*std::conj(z[k]);
				}
                
				// correct for sampling rate fa!=f, due to start of symbol being shifted ahead
				float cpi=2*PI*(d_delta/(float)d_Tu-d_Ndel/(float)d_Nfft);
				for(j=1;j<=d_K/2;j++){
					y[j]*=std::polar(1.0f,cpi*(j+ctr));
				}
				for(j=-d_K/2;j<=-1;j++){
					y[d_Nfft+j]*=std::polar(1.0f,cpi*(j+ctr));
				}
                
				// compensate for integer CFO due to CP
				for(j=0;j<d_Nfft;j++) {
					y[j]*=std::polar(1.0f,-2.0f*PI*d_delta/(float)d_Tu*ctr);
				}
                
				// frequency deinterleave
				for(j=0;j<d_K;j++) {
					gr_complex q=y[d_freqint.interleave(j)];
					
					out0[j]=std::real(q);
					out0[j+d_K]=std::imag(q);
				}
				nproduced+=2*d_K;
				out0+=2*d_K;
				if(i<=d_Nficsyms) { // FIC symbols
					memset(out1,1,sizeof(char)*2*d_K);
				}else { // MSC symbols
					memset(out1,0,sizeof(char)*2*d_K);
				}
				out1+=2*d_K;
			}
			// keep the current ofdm symbol for next symbol diff demod
			memcpy(z,x,sizeof(gr_complex)*d_Nfft);
		}
        nconsumed+=d_samples_to_consume_per_frame;
        in0+=d_samples_to_consume_per_frame;
        in1+=d_samples_to_consume_per_frame;
	}
	consume_each(nconsumed);
    return nproduced;
}
