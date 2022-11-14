/* -*- c++ -*- */
/* 
 * Copyright 2011 FOI
 *
 * Copyright 2010 A.Kaszuba, R.Checinski, MUT
 *
 * Copyright 2006,2007,2008,2010 Free Software Foundation, Inc.
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
// This is a modification of gr_ofdm_frame_acqusition.cc from GNU Radio.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_ofdm_alamouti_frame_acquisition.h>
#include <gr_io_signature.h>
#include <gr_expj.h>
#include <gr_math.h>
#include <stdexcept>
#include <iostream>
#include <string.h>
#include <cstdio>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

#define VERBOSE 0
#define M_TWOPI (2*M_PI)
#define MAX_NUM_SYMBOLS 1000

foimimo_ofdm_alamouti_frame_acquisition_sptr
foimimo_make_ofdm_alamouti_frame_acquisition (int ntxchannels, unsigned int occupied_carriers,
                                        unsigned int fft_length, unsigned int cplen,
                                        const std::vector<gr_complex> &known_symbol0,
                                        const std::vector<gr_complex> &known_symbol1,
                                        unsigned int max_fft_shift_len)
{
   return foimimo_ofdm_alamouti_frame_acquisition_sptr
     (new foimimo_ofdm_alamouti_frame_acquisition (ntxchannels, occupied_carriers, fft_length, cplen,
                                            known_symbol0, known_symbol1, max_fft_shift_len));
}

foimimo_ofdm_alamouti_frame_acquisition::foimimo_ofdm_alamouti_frame_acquisition (int ntxchannels,
                                                                       unsigned occupied_carriers,
                                                                       unsigned int fft_length,
                                                                       unsigned int cplen,
                                                                       const std::vector<gr_complex> &known_symbol0,
                                                                       const std::vector<gr_complex> &known_symbol1,
                                                                       unsigned int max_fft_shift_len)
 : gr_block ("ofdm_alamouti_frame_acquisition",
             gr_make_io_signature2 (3, 3, sizeof(char)*fft_length, sizeof(gr_complex)*fft_length),
             gr_make_io_signature2 (2, 2, sizeof(char), sizeof(gr_complex)*occupied_carriers)),
   d_occupied_carriers(occupied_carriers),
   d_fft_length(fft_length),
   d_cplen(cplen),
   d_freq_shift_len(max_fft_shift_len),
   d_known_symbol0(known_symbol0),
   d_known_symbol1(known_symbol1),
   d_phase_count(0)
{
   d_ntxchannels = ntxchannels;
   d_nrxchannels = 2;

   d_coarse_freq1 = new int[d_ntxchannels];
   d_coarse_freq2 = new int[d_ntxchannels];

   if(d_ntxchannels != 2)
     throw std::out_of_range("Alamouti Frame Acquisition: Can only have 2 antenna inputs.");

   d_hestimate1 = new std::vector<gr_complex>[d_ntxchannels];
   d_hestimate2 = new std::vector<gr_complex>[d_ntxchannels];
 
   d_snr_est = new float[d_ntxchannels];

   d_symbol_phase_diff1.resize(d_fft_length);
   d_symbol_phase_diff2.resize(d_fft_length);

   d_known_phase_diff1.resize(d_occupied_carriers);
   d_known_phase_diff2.resize(d_occupied_carriers);

   for(int i=0; i < d_ntxchannels; i++) {
     d_hestimate1[i].resize(d_occupied_carriers);
     d_hestimate2[i].resize(d_occupied_carriers);
   }
   for(int i=0; i < d_ntxchannels; i++) {
     for(unsigned int k=0; k<d_occupied_carriers; k++) {
     d_hestimate1[i][k] = gr_complex(1.0,0.0);
     d_hestimate2[i][k] = gr_complex(1.0,0.0);
     }
   }
   unsigned int i = 0, j = 0;

   std::fill(d_known_phase_diff1.begin(), d_known_phase_diff1.end(), 0);
   for(i = 0; i < d_known_symbol0.size()-1; i++) {
     d_known_phase_diff1[i] = norm(d_known_symbol0[i] - d_known_symbol0[i+1]);
   }
   i = 0;
 
   std::fill(d_known_phase_diff2.begin(), d_known_phase_diff2.end(), 0);
   for(i = 0; i < d_known_symbol1.size()-1; i++) {
     d_known_phase_diff2[i] = norm(d_known_symbol1[i] - d_known_symbol1[i+1]);
   }

   d_phase_lut = new gr_complex[(2*d_freq_shift_len+1) * MAX_NUM_SYMBOLS];
   for(i = 0; i <= 2*d_freq_shift_len; i++) {
     for(j = 0; j < MAX_NUM_SYMBOLS; j++) {
       d_phase_lut[j + i*MAX_NUM_SYMBOLS] =  gr_expj(-M_TWOPI*d_cplen/d_fft_length*(i-d_freq_shift_len)*j);
     }
   }
   first_preamble = false;
   second_preamble = false;
 }

foimimo_ofdm_alamouti_frame_acquisition::~foimimo_ofdm_alamouti_frame_acquisition(void)
{
   delete [] d_hestimate1;
   delete [] d_hestimate2;
   delete [] d_snr_est;
   delete [] d_phase_lut;
}

void
foimimo_ofdm_alamouti_frame_acquisition::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
   unsigned ninputs = ninput_items_required.size ();
   for (unsigned i = 0; i < ninputs; i++)
     ninput_items_required[i] = 1;
}

gr_complex
foimimo_ofdm_alamouti_frame_acquisition::coarse_freq_comp(int freq_delta, int symbol_count)
{
   return gr_expj(-M_TWOPI*freq_delta*d_cplen/d_fft_length*symbol_count);
}

void
foimimo_ofdm_alamouti_frame_acquisition::correlate1(const gr_complex *symbol0, const gr_complex *symbol1, int zeros_on_left)
{
   unsigned int i,j;

   std::fill(d_symbol_phase_diff1.begin(), d_symbol_phase_diff1.end(), 0);
   std::fill(d_symbol_phase_diff2.begin(), d_symbol_phase_diff2.end(), 0);
   for(i = 0; i < d_fft_length-1; i++) {
     d_symbol_phase_diff1[i] = norm(symbol0[i] - symbol0[i+1]);
     d_symbol_phase_diff2[i] = norm(symbol1[i] - symbol1[i+1]);
   }

   // sweep through all possible/allowed frequency offsets and select the best
   int index1 = 0;
   float max1 = 0, sum1=0;
   int index2 = 0;
   float max2 = 0, sum2=0;
   for(i =  zeros_on_left - d_freq_shift_len; i < zeros_on_left + d_freq_shift_len; i++) {
     sum1 = 0;
     sum2 = 0;
     
     for(j = 0; j < d_occupied_carriers; j++) {
       sum1 += (d_known_phase_diff1[j] * d_symbol_phase_diff1[i+j]);
       sum2 += (d_known_phase_diff1[j] * d_symbol_phase_diff2[i+j]);
     }
     
     if(sum1 > max1) {
       max1 = sum1;
       index1 = i;
     }
     if(sum2 > max2) {
       max2 = sum2;
       index2 = i;
     }
   }
   // set the coarse frequency offset relative to the edge of the occupied tones
   d_coarse_freq1[0] = index1 - zeros_on_left;
   d_coarse_freq2[0] = index2 - zeros_on_left;
}

void
foimimo_ofdm_alamouti_frame_acquisition::correlate2(const gr_complex *symbol0, const gr_complex *symbol1, int zeros_on_left)
{
  unsigned int i,j;

  std::fill(d_symbol_phase_diff1.begin(), d_symbol_phase_diff1.end(), 0);
  std::fill(d_symbol_phase_diff2.begin(), d_symbol_phase_diff2.end(), 0);
  for(i = 0; i < d_fft_length-1; i++) {
    d_symbol_phase_diff1[i] = norm(symbol0[i] - symbol0[i+1]);
    d_symbol_phase_diff2[i] = norm(symbol1[i] - symbol1[i+1]);
  }

  // sweep through all possible/allowed frequency offsets and select the best
  int index1 = 0;
  float max1 = 0, sum1=0;
  int index2 = 0;
  float max2 = 0, sum2=0;
  for(i =  zeros_on_left - d_freq_shift_len; i < zeros_on_left + d_freq_shift_len; i++) {
    sum1 = 0;
    sum2 = 0;

    for(j = 0; j < d_occupied_carriers; j++) {
      sum1 += (d_known_phase_diff2[j] * d_symbol_phase_diff1[i+j]);
      sum2 += (d_known_phase_diff2[j] * d_symbol_phase_diff2[i+j]);
    }
    
    if(sum1 > max1) {
      max1 = sum1;
      index1 = i;
    }
    if(sum2 > max2) {
      max2 = sum2;
      index2 = i;
    }
  }
  // set the coarse frequency offset relative to the edge of the occupied tones
  d_coarse_freq1[1] = index1 - zeros_on_left;
  d_coarse_freq2[1] = index2 - zeros_on_left;

}

void
foimimo_ofdm_alamouti_frame_acquisition::calculate_equalizer(int channel, const gr_complex *symbol1, const gr_complex *symbol2, int zeros_on_left)
{
   unsigned int i=0;

   // hack to work with different symbols; maybe replace d_known_symbols as a vector of vectors
   std::vector<gr_complex> preamble;
   if(channel == 0){
     preamble = d_known_symbol0;
     // Set first tap of equalizer
     d_hestimate1[channel][0] = (coarse_freq_comp(d_coarse_freq1[0],1)*symbol1[zeros_on_left+d_coarse_freq1[0]])/preamble[0];
     d_hestimate2[channel][0] = (coarse_freq_comp(d_coarse_freq2[0],1)*symbol2[zeros_on_left+d_coarse_freq2[0]])/preamble[0];
   
     for(i = 1; i < d_occupied_carriers; i++) {
       d_hestimate1[channel][i] =(coarse_freq_comp(d_coarse_freq1[0],1)*(symbol1[i+zeros_on_left+d_coarse_freq1[0]]))/preamble[i];
       d_hestimate2[channel][i] = (coarse_freq_comp(d_coarse_freq2[0],1)*(symbol2[i+zeros_on_left+d_coarse_freq2[0]]))/preamble[i];
     }

     // with even number of carriers; last equalizer tap is wrong
     if(!(d_occupied_carriers & 1)) {
       d_hestimate1[channel][d_occupied_carriers-1] = d_hestimate1[channel][d_occupied_carriers-2];
       d_hestimate2[channel][d_occupied_carriers-1] = d_hestimate2[channel][d_occupied_carriers-2];
     }
   } 
   else {
     preamble = d_known_symbol1;
  
     // Set first tap of equalizer
     d_hestimate1[channel][0] = (coarse_freq_comp(d_coarse_freq1[1],2)*symbol1[zeros_on_left+d_coarse_freq1[1]])/preamble[0];
     d_hestimate2[channel][0] = (coarse_freq_comp(d_coarse_freq2[1],2)*symbol2[zeros_on_left+d_coarse_freq2[1]])/preamble[0];

     // set every even tap based on known symbol
     // linearly interpolate between set carriers to set zero-filled carriers
     for(i = 1; i < d_occupied_carriers; i++) {
       d_hestimate1[channel][i] = (coarse_freq_comp(d_coarse_freq1[1],2)*(symbol1[i+zeros_on_left+d_coarse_freq1[1]]))/preamble[i];
       d_hestimate2[channel][i] = (coarse_freq_comp(d_coarse_freq2[1],2)*(symbol2[i+zeros_on_left+d_coarse_freq2[1]]))/preamble[i];
     }

   // with even number of carriers; last equalizer tap is wrong
     if(!(d_occupied_carriers & 1)) {
       d_hestimate1[channel][d_occupied_carriers-1] = d_hestimate1[channel][d_occupied_carriers-2];
       d_hestimate2[channel][d_occupied_carriers-1] = d_hestimate2[channel][d_occupied_carriers-2];
     }
   }
}

int
foimimo_ofdm_alamouti_frame_acquisition::general_work(int noutput_items,
                                                gr_vector_int &ninput_items,
                                                gr_vector_const_void_star &input_items,
                                                gr_vector_void_star &output_items)
{
   const char *signal_in = (const char *)input_items[0];
   const gr_complex *symbol1 = (const gr_complex *)input_items[1];
   const gr_complex *symbol2 = (const gr_complex *)input_items[2];

 
   char *signal_out = (char *) output_items[0]; 
   gr_complex *out = (gr_complex *) output_items[1];
 
   int unoccupied_carriers = d_fft_length - d_occupied_carriers;
   int zeros_on_left = (int)ceil(unoccupied_carriers/2.0);

   signal_out[0] = 0;

   // Test if we hit the start of a preamble
   if(signal_in[0]) {
     first_preamble = true;
   }
   else if(first_preamble){
     first_preamble = false;
     d_phase_count = 1;

     correlate1(&symbol1[0], &symbol2[0], zeros_on_left);
     printf("Coarse freq1[0]:%d Coarse freq2[0]:%d \n",d_coarse_freq1[0],d_coarse_freq2[0]);
     
     calculate_equalizer(0, &symbol1[0], &symbol2[0], zeros_on_left);
   
     signal_out[0] = 1;
     second_preamble = true;

     // equalize the 1st preamble
     for(unsigned int i = 0; i < d_occupied_carriers; i+=2) {
         out[i] = (coarse_freq_comp(d_coarse_freq1[0],d_phase_count)
           *symbol1[i+zeros_on_left+d_coarse_freq1[0]])/d_hestimate1[0][i];

        out[i+1] = (coarse_freq_comp(d_coarse_freq1[0],d_phase_count)
           *symbol1[i+1+zeros_on_left+d_coarse_freq1[0]])/d_hestimate1[0][i+1];
       }
     
   }
   else if(second_preamble) {
     correlate2(&symbol1[0], &symbol2[0], zeros_on_left);
     printf("Coarse freq1[1]:%d Coarse freq2[1]:%d \n",d_coarse_freq1[1],d_coarse_freq2[1]);
     calculate_equalizer(1, &symbol1[0], &symbol2[0], zeros_on_left);
     second_preamble = false;

     //equalize the 2nd preamble
     for(unsigned int i = 0; i < d_occupied_carriers; i+=2) {
         out[i] = (coarse_freq_comp(d_coarse_freq2[1],d_phase_count)
           *symbol2[i+zeros_on_left+d_coarse_freq2[1]])/d_hestimate2[1][i];

         out[i+1] = (coarse_freq_comp(d_coarse_freq2[1],d_phase_count)
           *symbol2[i+1+zeros_on_left+d_coarse_freq2[1]])/d_hestimate2[1][i+1];
       }
     

   }
   else{
     // equalize each OFDM-symbol
     for(unsigned int i = 0; i < d_occupied_carriers/2; i++) {
       gr_complex y00 = symbol1[2*i+zeros_on_left + d_coarse_freq1[0]]* coarse_freq_comp(d_coarse_freq1[0],d_phase_count) ; //y1
       gr_complex y10 = symbol1[2*i+1+zeros_on_left + d_coarse_freq1[0]] * coarse_freq_comp(d_coarse_freq1[0],d_phase_count) ; //y2
       gr_complex y01 = symbol2[2*i +zeros_on_left + d_coarse_freq2[1]] * coarse_freq_comp(d_coarse_freq2[1],d_phase_count) ; //y3
       gr_complex y11 = symbol2[2*i+1 +zeros_on_left + d_coarse_freq2[1]] * coarse_freq_comp(d_coarse_freq2[1],d_phase_count); //y4

       //Alamouti combining SFBC
       gr_complex out1 = (conj(d_hestimate1[0][2*i])*y00 + d_hestimate1[1][2*i+1]*conj(y10) + conj(d_hestimate2[0][2*i])*y01 + d_hestimate2[1][2*i+1]*conj(y11)); //s1^

       gr_complex out2 = (conj(d_hestimate1[1][2*i])*y00 - d_hestimate1[0][2*i+1]*conj(y10) + conj(d_hestimate2[1][2*i])*y01 - d_hestimate2[0][2*i+1]*conj(y11)); //s2^

       out[2*i] = out1;
       out[2*i+1] = out2;
     }
   }

   
   d_phase_count++;
   if(d_phase_count >= MAX_NUM_SYMBOLS) {
     d_phase_count = 1;
   }
 
   consume_each(1);
   return 1;
}
