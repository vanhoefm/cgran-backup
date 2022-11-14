/* -*- c++ -*- */
/*
 * Copyright 2004,2006 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
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


// This block receives in input the IQ signal, calculates the magnitude of each sample (ASK demodulator) and sends it to 3 different output (one for the rx reader chain, one for the rx tag chain and one for the carrier tracking chain)

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <listener_to_mag_mux.h>
#include <gr_io_signature.h>
#include <stdlib.h>
#include <stdio.h>
#include <gr_math.h>


/////////////////////////////////////////////////////////////////
// INITIAL SETUP						/
/////////////////////////////////////////////////////////////////
listener_to_mag_mux_sptr
listener_make_to_mag_mux ()
{
	return listener_to_mag_mux_sptr (new listener_to_mag_mux ());
}

listener_to_mag_mux::listener_to_mag_mux()
  : gr_block("listener_to_mag_mux", gr_make_io_signature (1, 1, sizeof(gr_complex)), gr_make_io_signature3 (3,3,sizeof(float),sizeof(float),sizeof(gr_complex)))
 
{
}
// END INITIAL SETUP
//////////////////////////////////////////////////////////////////////////////////////////////////

 
listener_to_mag_mux::~listener_to_mag_mux()
{
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// GENERAL WORK										 /
//////////////////////////////////////////////////////////////////////////////////////////////////
int listener_to_mag_mux::general_work(int noutput_items,
			   gr_vector_int &ninput_items,
			   gr_vector_const_void_star &input_items,
			   gr_vector_void_star &output_items)
{

	const gr_complex * in = (const gr_complex *)input_items[0];
	float            * out_0 = (float * )output_items[0];
	float            * out_1 = (float * )output_items[1];
	gr_complex       * out_2 = (gr_complex * )output_items[2];
	float mag;

  	for (int i = 0; i < noutput_items; i++) {
		mag = std::abs (in[i]);
		out_0[i] = out_1[i] = mag;
		out_2[i] = in[i];
	} //END FOR
 
	consume_each(noutput_items);
	return noutput_items;
 
}
// END GENERAL WORK
////////////////////////////////////////////////////////////////////////////////////////// 



//////////////////////////////////////////////////////////////////////////////////////////
// FORECAST										 /
//////////////////////////////////////////////////////////////////////////////////////////
void listener_to_mag_mux::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
	unsigned ninputs = ninput_items_required.size ();
	for (unsigned i = 0; i < ninputs; i++){
		ninput_items_required[i] = noutput_items;
	}   
}
// END FORECAST
//////////////////////////////////////////////////////////////////////////////////////////
