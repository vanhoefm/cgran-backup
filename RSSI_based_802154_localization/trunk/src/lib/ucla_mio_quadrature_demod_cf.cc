/* -*- c++ -*- */
/*
 * Copyright 2004,2005 Free Software Foundation, Inc.
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
*
Modified by Angelo Coluccia 
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucla_mio_quadrature_demod_cf.h>
#include <gr_io_signature.h>
#include <gr_math.h>
#include <cstdio>

float RSSI;  //global variable

ucla_mio_quadrature_demod_cf::ucla_mio_quadrature_demod_cf (float gain)
  : gr_sync_block ("mio_quadrature_demod_cf",
		   gr_make_io_signature (1, 1, sizeof (gr_complex)),
		   gr_make_io_signature (1, 1, sizeof (float))),
    d_gain (gain)
{
  set_history (2);	// we need to look at the previous value
}

ucla_mio_quadrature_demod_cf_sptr
ucla_make_mio_quadrature_demod_cf (float gain)
{
  return ucla_mio_quadrature_demod_cf_sptr (new ucla_mio_quadrature_demod_cf (gain));
}

int
ucla_mio_quadrature_demod_cf::work (int noutput_items,
				   gr_vector_const_void_star &input_items,
				   gr_vector_void_star &output_items)
{
  gr_complex *in = (gr_complex *) input_items[0];
  float *out = (float *) output_items[0];
  in++;				// ensure that in[-1] is valid
  
float RSS=0;  
for (int i=0; i<16; i++) RSS+=abs(in[i]);

RSSI=RSS/16; // global variable

//fprintf(stderr,"%f\n",RSSI), fflush(stderr);

  for (int i = 0; i < noutput_items; i++){
    gr_complex product = in[i] * conj (in[i-1]);
    // out[i] = d_gain * arg (product);
    out[i] = d_gain * gr_fast_atan2f(imag(product), real(product));
  }

  return noutput_items;
}
