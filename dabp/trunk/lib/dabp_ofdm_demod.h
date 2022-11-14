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
#ifndef INCLUDED_DABP_OFDM_DEMOD_H
#define INCLUDED_DABP_OFDM_DEMOD_H
#include <gr_block.h>
#include <fftw3.h>
#include <cstdio>
#include "dabp_freq_interleaver.h"
#include "dabp_parameters.h"

class dabp_ofdm_demod;

typedef boost::shared_ptr<dabp_ofdm_demod> dabp_ofdm_demod_sptr;

dabp_ofdm_demod_sptr dabp_make_ofdm_demod(int mode);

/*!
 * \brief OFDM demodulator
 * \ingroup ofdm_demod
 * Two input streams: complex samples and indicator of Null symbol
 * Two output streams: float demodulated samples and indicator of FIC symbols
 * The output of the demodulated samples output is organized as one DAB frame at a time
 * i.e. L-1 ofdm symbols at a time. The first Nficsyms are FIC and the rest are MSC
 */
class dabp_ofdm_demod : public gr_block
{
    private:
    friend dabp_ofdm_demod_sptr dabp_make_ofdm_demod(int mode);
    dabp_ofdm_demod(int mode);
    dabp_parameters d_param;
    int d_L, d_Tu, d_delta, d_K, d_Tnull;
    float d_f, d_fa;
	int d_Ts, d_Ndel, d_Nfft, d_Nfrm, d_Nnull;
    int d_Nficsyms;
	dabp_freq_interleaver d_freqint;
	static const int TSYNC_TOL=10;
	static const int MAXSC=10;
	static const float PI=3.14159265;
	fftwf_complex *d_fftin, *d_fftout;
	fftwf_plan d_plan;
	
    int d_samples_to_consume_per_frame;
    
    public:
    ~dabp_ofdm_demod();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};

#endif // INCLUDED_DABP_OFDM_DEMOD_H

