/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
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
// This is a modification of trellis_metrics_f.cc from GNU Radio.


#ifndef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_trellis_metrics_f.h>
#include <gr_io_signature.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <math.h>
#include <stdio.h>


foimimo_trellis_metrics_f_sptr
foimimo_make_trellis_metrics_f (int O, int D, const std::vector<float> &REtable,
    const std::vector<float> &IMtable)
{
  return foimimo_trellis_metrics_f_sptr (new foimimo_trellis_metrics_f (O,D,REtable,IMtable));
}

foimimo_trellis_metrics_f::foimimo_trellis_metrics_f (int O, int D,  const std::vector<float> &REtable,
                                          const std::vector<float> &IMtable)
  : gr_block ("metrics_f",
	      gr_make_io_signature2 (2, 2, sizeof (float), sizeof(char)),
	      gr_make_io_signature2 (2, 2, sizeof (float), sizeof(char))),
    d_O (O),
    d_D (D),
    d_re(REtable),
    d_im(IMtable),
    d_real_part(true)
{
  set_relative_rate (d_O / (double) log2(d_O));
  set_output_multiple (d_O);
}

void
foimimo_trellis_metrics_f::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  assert (noutput_items % d_O == 0);
  int input_required =  ceil(d_D * log2(d_O) * noutput_items / (2.0 * d_O));
  // log2(d_O)/2  2: We have QPSK for now.

  unsigned ninputs = ninput_items_required.size();
  for (unsigned int i = 0; i < ninputs; i++)
    ninput_items_required[i] = input_required;
}



void
foimimo_trellis_metrics_f::calc_metric(const float *in,float *metric){
  //Fixme Does only support QPSK. that's the reason for the constant 2 in this function
  int bits_in_codeword = ceil(log2(d_O));
  std::vector<float> distance (2*bits_in_codeword,0.0);
  bool re = d_real_part;

  // Calculate distance
  int in_cnt=0;
  for(int i=0;i<distance.size();i+=2){
    if(re){
      distance[i]   = pow((d_re[0]-in[in_cnt]),2);
      distance[i+1] = pow((d_re[1]-in[in_cnt]),2);
    }else{
      distance[i]   = pow((d_im[0]-in[in_cnt]),2);
      distance[i+1] = pow((d_im[1]-in[in_cnt]),2);
    }
    in_cnt++;
    re = !re;
  }

  // sum up the distance to a metric for all possible output symbols
  int k=0;
  for(int i=0;i<d_O;i++){
    k=0;
    metric[i]=0.0;
    for(int j=0;j<distance.size();j+=2,k++){
      metric[i] += distance[j+((i>>k) & 0x1)];
    }
  }

}

int
foimimo_trellis_metrics_f::general_work (int noutput_items,
				gr_vector_int &ninput_items,
				gr_vector_const_void_star &input_items,
				gr_vector_void_star &output_items)
{

  assert (noutput_items % d_O == 0);

  const float *in = (float *) input_items[0];
  unsigned char *in_new_pkt = (unsigned char *) input_items[1];
  float *out = (float *) output_items[0];
  unsigned char *out_new_pkt = (unsigned char *) output_items[1];

  int iptr = 0;
  int optr = 0;

  int bits_in_codeword = (int) (ceil(log2(d_O)));

  while(iptr+bits_in_codeword <= ninput_items[0] && optr + d_O <= noutput_items){
    calc_metric(&in[iptr],&out[optr]);
    memset(&out_new_pkt[optr],0,d_O);
    //check if it's a new packet coming
    if(iptr == 0){
      // We have a new packet
      if (in_new_pkt[iptr] == 1){
        out_new_pkt[optr] = 1;
      }else{
        for(int i=1; i < bits_in_codeword;i++){
          // we have a new packet coming return this packet and start over with the next
          if (in_new_pkt[iptr+i] == 1){
            d_real_part = true;
            consume_each(iptr+i);
            return optr;
          }
        }
      }
    }else{
      for(int i=0; i < bits_in_codeword;i++){
        // we have a new packet coming return this packet and start over with the next
        if (in_new_pkt[iptr+i] == 1){
          d_real_part = true;
          consume_each(iptr+i);
          return optr;
        }
      }
    }
    // Toggle the real part info for codewords with odd number of bits
    if (bits_in_codeword%2 == 1){
      d_real_part = !d_real_part;
    }
    iptr += bits_in_codeword;
    optr += d_O;
  }
  consume_each(iptr);
  return optr;
}
