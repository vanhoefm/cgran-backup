/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
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

// @WARNING@

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <@NAME@.h>
#include <gr_io_signature.h>
#include <string.h>
#include <math.h> //linklab

@SPTR_NAME@
gr_make_@BASE_NAME@ (float threshold_factor_rise,
		     float threshold_factor_fall,
		     int look_ahead, float alpha, int carrier_num)
{
  return @SPTR_NAME@ (new @NAME@ (threshold_factor_rise, 
				  threshold_factor_fall,
				  look_ahead, alpha, carrier_num));
}

@NAME@::@NAME@ (float threshold_factor_rise, 
		float threshold_factor_fall,
		int look_ahead, float alpha, int carrier_num)
  : gr_sync_block ("@BASE_NAME@",
		   gr_make_io_signature2 (2, 2, sizeof (float), sizeof (@I_TYPE@)),
		   gr_make_io_signature (1, 1, sizeof (char))),
    d_threshold_factor_rise(threshold_factor_rise), 
    d_threshold_factor_fall(threshold_factor_fall),
    d_look_ahead(look_ahead), d_avg_alpha(alpha), 
    d_avg(-1), // linklab
    d_sinr(0), // linklab
    d_found(0),
    d_carrier_num(carrier_num)
{
}

int
@NAME@::work (int noutput_items,
	      gr_vector_const_void_star &input_items,
	      gr_vector_void_star &output_items)
{
  @I_TYPE@ *iptr = (@I_TYPE@ *) input_items[0];
  float *ippw = (float *) input_items[1];// linklab, input sig power
  char *optr = (char *) output_items[0];

  memset(optr, 0, noutput_items*sizeof(char));

  @I_TYPE@ peak_val = -(@I_TYPE@)INFINITY;
  int peak_ind = 0;
  unsigned char state = 0;
  int i = 0;

  //printf("noutput_items %d\n",noutput_items);
  while(i < noutput_items) {
    d_sig_power = sqrt(ippw[i]); // linklab, get signal power
    if(state == 0) {  // below threshold
      if(iptr[i] > d_avg*d_threshold_factor_rise) {
	state = 1;
      }
      else {
	d_avg = (d_avg_alpha)*iptr[i] + (1-d_avg_alpha)*d_avg;
	i++;
      }
    }
    else if(state == 1) {  // above threshold, have not found peak
      //printf("Entered State 1: %f  i: %d  noutput_items: %d\n", iptr[i], i, noutput_items);
      if(iptr[i] > peak_val) {
	peak_val = iptr[i];
	peak_ind = i;
	d_avg = (d_avg_alpha)*iptr[i] + (1-d_avg_alpha)*d_avg;
	i++;
      }
      // linklab, add the leaked peak value near the end and wipe out the false peak value at the start
      // else if (iptr[i] > d_avg*d_threshold_factor_fall) {
      else if ((iptr[i] > d_avg*d_threshold_factor_fall) && (i < noutput_items-1) || (peak_ind == 0)){
        d_avg = (d_avg_alpha)*iptr[i] + (1-d_avg_alpha)*d_avg;
        i++;
      }
      else {
          optr[peak_ind] = 1;
          // calculate the sinr in time domain, linklab
          if (peak_val < 0)
              d_sinr = 10*log(1/(1/(sqrt(peak_val+1))-1))/log(10);
          else
              d_sinr = (@I_TYPE@)INFINITY; 

	  state = 0;
	  peak_val = -(@I_TYPE@)INFINITY;
	  //printf("Leaving  State 1: Peak: %f  Peak Ind: %d   i: %d  noutput_items: %d\n", 
	  //peak_val, peak_ind, i, noutput_items);
      }
    }
  }

  if(state == 0) {
    //printf("Leave in State 0, produced %d\n",noutput_items);
    return noutput_items;
  }
  else {   // only return up to passing the threshold
    //printf("Leave in State 1, only produced %d of %d\n",peak_ind,noutput_items);
    return peak_ind+1;
  }
}
