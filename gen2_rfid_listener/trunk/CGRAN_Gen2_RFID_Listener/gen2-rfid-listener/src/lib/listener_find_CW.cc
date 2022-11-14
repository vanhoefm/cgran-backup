/* -*- c++ -*- */
/*
 * Copyright 2006 Free Software Foundation, Inc.
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

#include <listener_find_CW.h>
#include <gr_io_signature.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <listener_reader_monitor_cmd_gate.h>


// This block is responsible of determining where the most powerful signal (reader transmission) is located over the acquired band.
// It has 4 input parameters: the fft size, the USRP center frequency, the sampling rate and the reader monitor block to update the new frequency.

listener_find_CW_sptr
listener_make_find_CW(unsigned int vlen,
			float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor
			)
{
  return listener_find_CW_sptr(new listener_find_CW(vlen, usrp_freq, samp_rate, reader_monitor));
}

listener_find_CW::listener_find_CW(unsigned int vlen, float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor)
  : gr_sync_block("find_CW",
		  gr_make_io_signature(1, 1, sizeof(float) *vlen),
		  gr_make_io_signature(0, 0, 0)),
    d_vlen(vlen), d_usrp_freq(usrp_freq), d_samp_rate(samp_rate), d_reader_monitor(reader_monitor) 
{
max_bin_index_new=0;
max_bin_index_old=0;
max_abs=500; // this is a threshold to distinguish when we have noise or a reader signal. 
}


listener_find_CW::~listener_find_CW()
{
}


int listener_find_CW::work(int noutput_items,
			   gr_vector_const_void_star &input_items,
			   gr_vector_void_star &output_items)
{

  const float *input = (const float *) input_items[0];
    
  for (int i=0; i<d_vlen; i++) {
	d_max[i] = input[i];
  }
	
  max_bin_index_new = max_min(d_max, d_vlen);

  // This statement is an approssimative control to avoid of sending too many stats. The stat is sent only if I am quite sure that the reader has
  // changed its frequency. 
  if (max_bin_index_new!=9999) {
	if (max_bin_index_new != max_bin_index_old) {
		send_stat(max_bin_index_new);
  		max_bin_index_old = max_bin_index_new;
	}
  }
    
  return noutput_items;

}


// Update reader frequency in reader_monitor_cmd_gate block
void listener_find_CW::send_stat(int max_bin_index) {
	
	if (max_bin_index < d_vlen/2) {
		reader_freq = max_bin_index*(d_samp_rate/d_vlen);
	}
	else {
		reader_freq = (float)((max_bin_index-d_vlen/2)*(d_samp_rate/d_vlen))-(d_samp_rate/2.);
	}
	
	reader_freq = (d_usrp_freq+reader_freq)*1E-6;
		
	//printf("reader_freq = %.1f MHz\n",reader_freq);
	d_reader_monitor->set_reader_freq(reader_freq);
  
}


// Determine max FFT bin
int listener_find_CW::max_min(float * buffer, int len) {

  float max_tmp = buffer[0];
  int max_bin=9999;

  for (size_t i = 0; i < len; i++){
  	if(buffer[i] >= max_tmp) {
       		max_tmp = buffer[i];
       		if (max_tmp>max_abs/2) {max_abs=max_tmp; max_bin = i;}
       }
  }
  
  return max_bin;

}


