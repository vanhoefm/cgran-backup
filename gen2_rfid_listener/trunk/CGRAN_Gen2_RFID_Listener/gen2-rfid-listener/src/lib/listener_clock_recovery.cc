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


// Acknowledgement: I would like to acknowledge Michael Buettner since this adapted and modified block is inherited from his CGRAN project "Gen2 RFID Reader"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <listener_clock_recovery.h>
#include <gr_io_signature.h>
#include <gri_mmse_fir_interpolator.h>
#include <stdexcept>
#include <float.h>
#include <string.h>

#ifndef TAG_MONITOR_VARS
#include "tag_monitor_vars.h"
#endif

// EXECUTED OPERATIONS
// 1. Interpolate the signal by interp_factor
// 2. Center the signal at 0 amplitude
// 3. Find the zero crossings
// 4. Sample at the appropriate distance from the zero crossing


/////////////////////////////////////////////////////////////////
// INITIAL SETUP
////////////////////////////////////////////////////////////////
listener_clock_recovery_sptr
listener_make_clock_recovery(int samples_per_pulse, int interp_factor)
{
  return listener_clock_recovery_sptr(new listener_clock_recovery(samples_per_pulse, interp_factor));
}
listener_clock_recovery::listener_clock_recovery(int samples_per_pulse, int interp_factor)
  : gr_block("listener_clock_recovery", 
		      gr_make_io_signature(1,1,sizeof(float)),
		      gr_make_io_signature(1,1,sizeof(float))),
    d_samples_per_pulse(samples_per_pulse), d_interp_factor(interp_factor), 
    d_interp(new gri_mmse_fir_interpolator())
{

	set_history(d_interp->ntaps());

	d_interp_buffer = (float * )malloc(8196 * sizeof(float) * d_interp_factor);  //buffer for storing interpolated signal

	for(int i = 0; i < 8196 * d_interp_factor; i++){
		d_interp_buffer[i] = 0;  
	}

	d_last_zc_count = 0; 
	d_pwr = 0;

	int num_pulses = 16;  

	d_avg_window_size = d_samples_per_pulse * num_pulses; 
	d_last_was_pos = true;  

	d_avg_vec_index = 0;
	d_avg_vec = (float*)malloc(d_avg_window_size * sizeof(float));
	for(int i = 0; i < d_avg_window_size; i++){
		d_avg_vec[i] = 0;  
	}

}
// END INITIAL SETUP
//////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////
// FORECAST
//////////////////////////////////////////////////////////////////////////////////////////
void listener_clock_recovery::forecast(int noutput_items, gr_vector_int &ninput_items_required){
	unsigned ninputs = ninput_items_required.size ();
	for (unsigned i = 0; i < ninputs; i++){
		ninput_items_required[i] = noutput_items + history();
	}   
}
// END FORECAST
////////////////////////////////////////////////////////////////////////////////////////// 



listener_clock_recovery::~listener_clock_recovery()
{
  delete d_interp;
}

static inline bool
is_positive(float x){
  return x < 0 ? false : true;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// GENERAL WORK
//////////////////////////////////////////////////////////////////////////////////////////////////
int listener_clock_recovery::general_work(int noutput_items,
					gr_vector_int &ninput_items,
					gr_vector_const_void_star &input_items,
					gr_vector_void_star &output_items)
{

	const float *in = (const float *) input_items[0];
	float* out = (float *) output_items[0];
	int debug_out_cnt1 = 0;
	int debug_out_cnt2 = 0;
	int nout = 0;
	int num_past_samples = d_samples_per_pulse * d_interp_factor;  	
  	int num_interp_samples = 0;


  	for(int i = 0; i < noutput_items; i++){ // Interpolate and center signal 
                                           	
		//Calculate average
		d_pwr -= d_avg_vec[d_avg_vec_index];
		d_pwr += in[i];
		d_avg_vec[d_avg_vec_index++] = in[i];
        
		if(d_avg_vec_index == d_avg_window_size){
			d_avg_vec_index = 0;
		}

		for(int j = 0; j < d_interp_factor; j++){
			d_interp_buffer[(i * d_interp_factor) + j + num_past_samples] = (d_interp->interpolate(&in[i], ((1 / (float)d_interp_factor) * (float)j))) - (d_pwr / (float)d_avg_window_size);
			num_interp_samples++;
		}
    
	}

	//Find zero crossings, reduce sample rate by taking only the samples we need
	// Start after the num_past_samples worth of padding
	for(int i = num_past_samples; i < num_interp_samples + num_past_samples; i++){
		if((d_last_was_pos && ! is_positive(d_interp_buffer[i])) || (!d_last_was_pos && is_positive(d_interp_buffer[i]))){
		//We found a zero crossing, "look back" and take the sample from the middle of the last pulse. 
		// A long period between zero crossings indicates the long pulse of the miller encoding, 
		// so take two samples from center of pulse
			if(d_last_zc_count > (d_samples_per_pulse * d_interp_factor) * 1.25){
				out[nout++] = d_interp_buffer[i - (d_last_zc_count / 2)];
				out[nout++] = d_interp_buffer[i - (d_last_zc_count / 2)];
			}
			else{
				out[nout++] = d_interp_buffer[i - (d_last_zc_count / 2)];
			}
			d_last_zc_count = 0;
		}
		else{
			d_last_zc_count++;
		}
		
		d_last_was_pos = is_positive(d_interp_buffer[i]);

	}

	//Copy num_past_samples to head of buffer so we can "look back" during the next general_work call
	memcpy(d_interp_buffer, &d_interp_buffer[num_interp_samples], num_past_samples * sizeof(float));

	consume_each(noutput_items);

	return nout;

}
// END GENERAL WORK
//////////////////////////////////////////////////////////////////////////////////////////       
