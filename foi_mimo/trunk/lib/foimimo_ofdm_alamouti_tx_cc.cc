/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2010 A.Kaszuba, R.Checinski, MUT
 *
 * This file is part of FOI-MIMO
 * 
 * FOI-MIMO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * FOI-MIMO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FOI-MIMO; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif 

#include <foimimo_ofdm_alamouti_tx_cc.h>
#include <gr_io_signature.h>
#include <stdexcept>
#include <iostream>
#include <string.h>
#include <cstdio>

foimimo_ofdm_alamouti_tx_cc_sptr
foimimo_make_ofdm_alamouti_tx_cc(int fft_length)
{
  return foimimo_ofdm_alamouti_tx_cc_sptr(new foimimo_ofdm_alamouti_tx_cc(fft_length));
}

foimimo_ofdm_alamouti_tx_cc::foimimo_ofdm_alamouti_tx_cc(int fft_length)
  : gr_sync_block ("ofdm_alamouti_tx_cc",
		  gr_make_io_signature(1, 1, sizeof(gr_complex)*fft_length),
		  gr_make_io_signature(2, 2, sizeof(gr_complex)*fft_length)),
    d_fft_length(fft_length)
{
}

foimimo_ofdm_alamouti_tx_cc::~foimimo_ofdm_alamouti_tx_cc()
{
}

int
foimimo_ofdm_alamouti_tx_cc::work (int noutput_items,
			      gr_vector_const_void_star &input_items,
			      gr_vector_void_star &output_items)
{
  const gr_complex *in = (const gr_complex *) input_items[0];
  gr_complex *out_sym0 = (gr_complex *) output_items[0];
  gr_complex *out_sym1 = (gr_complex *) output_items[1];

  int iptr = 0, optr = 0;
  while(iptr < noutput_items) {
 
    //SFBC:
    for(int j = 0; j < d_fft_length/2; j++) {
       out_sym0[optr*d_fft_length+2*j] = in[iptr*d_fft_length + 2*j];
       out_sym0[optr*d_fft_length+2*j + 1] = -conj(in[iptr*d_fft_length + 2*j + 1]);
    }
    for(int j = 0; j < d_fft_length/2; j++) {
       out_sym1[optr*d_fft_length + 2*j] = in[iptr*d_fft_length + 2*j + 1];
       out_sym1[optr*d_fft_length + 2*j + 1] = conj(in[iptr*d_fft_length + 2*j]);
    }
    iptr++;
    optr++;
  }
  return optr;
}
