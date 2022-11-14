/* -*- c++ -*- */
/*
 * Copyright 2006,2007,2008 Free Software Foundation, Inc.
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

#include <gr_ofdm_frame_acquisition.h>
#include <gr_io_signature.h>
#include <gr_expj.h>
#include <gr_math.h>
#include <cstdio>

#define VERBOSE 0
#define M_TWOPI (2*M_PI)
#define MAX_NUM_SYMBOLS 1000

gr_ofdm_frame_acquisition_sptr
gr_make_ofdm_frame_acquisition (unsigned int occupied_carriers, unsigned int fft_length, 
				unsigned int cplen,
				const std::vector<gr_complex> &known_symbol,
				unsigned int max_fft_shift_len)
{
  return gr_ofdm_frame_acquisition_sptr (new gr_ofdm_frame_acquisition (occupied_carriers, fft_length, cplen,
									known_symbol, max_fft_shift_len));
}

gr_ofdm_frame_acquisition::gr_ofdm_frame_acquisition (unsigned occupied_carriers, unsigned int fft_length, 
						      unsigned int cplen,
						      const std::vector<gr_complex> &known_symbol,
						      unsigned int max_fft_shift_len)
  : gr_block ("ofdm_frame_acquisition",
	      gr_make_io_signature2 (2, 2, sizeof(gr_complex)*fft_length, sizeof(char)*fft_length),
	      gr_make_io_signature2 (2, 2, sizeof(gr_complex)*occupied_carriers, sizeof(char))),
    d_occupied_carriers(occupied_carriers),
    d_fft_length(fft_length),
    d_cplen(cplen),
    d_freq_shift_len(max_fft_shift_len),
    d_known_symbol(known_symbol),
    d_coarse_freq(0),
    d_phase_count(0),
	d_sinr(0)
{
  d_known_symbol_tmp.clear();
  d_known_symbol_tmp.assign(d_known_symbol.begin(), d_known_symbol.end());
  initialize();
}

gr_ofdm_frame_acquisition::~gr_ofdm_frame_acquisition(void)
{
  delete [] d_phase_lut;
}

// linklab, initialization function to set parameters
void 
gr_ofdm_frame_acquisition::initialize()
{
  d_symbol_phase_diff.resize(d_fft_length);
  d_known_phase_diff.resize(d_occupied_carriers);
  d_hestimate.resize(d_occupied_carriers);
  d_ch = "";
  unsigned int i = 0, j = 0;

  std::fill(d_known_phase_diff.begin(), d_known_phase_diff.end(), 0);
  for(i = 0; i < d_known_symbol_tmp.size()-2; i+=2) {
    d_known_phase_diff[i] = norm(d_known_symbol_tmp[i] - d_known_symbol_tmp[i+2]);
  }
  
  d_phase_lut = new gr_complex[(2*d_freq_shift_len+1) * MAX_NUM_SYMBOLS];
  for(i = 0; i <= 2*d_freq_shift_len; i++) {
    for(j = 0; j < MAX_NUM_SYMBOLS; j++) {
      d_phase_lut[j + i*MAX_NUM_SYMBOLS] =  gr_expj(-M_TWOPI*d_cplen/d_fft_length*(i-d_freq_shift_len)*j);
    }
  }
}

void
gr_ofdm_frame_acquisition::reset_known_symbol(std::vector<gr_complex> new_known_symbol)
{
  d_known_symbol_tmp.clear();
  d_known_symbol_tmp.assign(new_known_symbol.begin(), new_known_symbol.end());
  initialize();
}

void
gr_ofdm_frame_acquisition::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  unsigned ninputs = ninput_items_required.size ();
  for (unsigned i = 0; i < ninputs; i++)
    ninput_items_required[i] = 1;
}

gr_complex
gr_ofdm_frame_acquisition::coarse_freq_comp(int freq_delta, int symbol_count)
{
  //  return gr_complex(cos(-M_TWOPI*freq_delta*d_cplen/d_fft_length*symbol_count),
  //	    sin(-M_TWOPI*freq_delta*d_cplen/d_fft_length*symbol_count));

  return gr_expj(-M_TWOPI*freq_delta*d_cplen/d_fft_length*symbol_count);

  //return d_phase_lut[MAX_NUM_SYMBOLS * (d_freq_shift_len + freq_delta) + symbol_count];
}

void
gr_ofdm_frame_acquisition::correlate(const gr_complex *symbol, int zeros_on_left)
{
  unsigned int i,j;
  
  std::fill(d_symbol_phase_diff.begin(), d_symbol_phase_diff.end(), 0);
  for(i = 0; i < d_fft_length-2; i++) {
    d_symbol_phase_diff[i] = norm(symbol[i] - symbol[i+2]);
  }

  // linklab, calc frequency shift 
  int freq_shift_len = 0;
  if (zeros_on_left < d_freq_shift_len)
    freq_shift_len = zeros_on_left;
  else
    freq_shift_len = d_freq_shift_len;
                                                         

  // sweep through all possible/allowed frequency offsets and select the best
  int index = 0;
  float max = 0, sum=0;
  //for(i =  zeros_on_left - d_freq_shift_len; i < zeros_on_left + d_freq_shift_len; i++) {  // replaced by the following
  for(i =  zeros_on_left - freq_shift_len; i < zeros_on_left + freq_shift_len; i++) {
    sum = 0;
    for(j = 0; j < d_occupied_carriers; j++) {
      sum += (d_known_phase_diff[j] * d_symbol_phase_diff[i+j]);
    }
    if(sum > max) {
      max = sum;
      index = i;
    }
  }
  // linklab 
  // set the coarse frequency offset relative to the edge of the occupied tones
  d_coarse_freq = index - zeros_on_left;
}

void
gr_ofdm_frame_acquisition::calculate_equalizer(const gr_complex *symbol, int zeros_on_left)
{
  unsigned int i=0;

  // Set first tap of equalizer
  d_hestimate[0] = d_known_symbol_tmp[0] / 
    (coarse_freq_comp(d_coarse_freq,1)*symbol[zeros_on_left+d_coarse_freq]);

  // set every even tap based on known symbol
  // linearly interpolate between set carriers to set zero-filled carriers
  // FIXME: is this the best way to set this?
  for(i = 2; i < d_occupied_carriers; i+=2) {
    d_hestimate[i] = d_known_symbol_tmp[i] / 
      (coarse_freq_comp(d_coarse_freq,1)*(symbol[i+zeros_on_left+d_coarse_freq]));
    
    //d_hestimate[i-1] = (d_hestimate[i] + d_hestimate[i-2]) / gr_complex(2.0, 0.0);    
    // linklab, adjust the channel estimated at boundaries 
    if (d_known_symbol_tmp[i].real()==0 || d_known_symbol_tmp[i-2].real()==0)
      d_hestimate[i-1] = d_hestimate[i] + d_hestimate[i-2];    // linklab
    else
      d_hestimate[i-1] = (d_hestimate[i] + d_hestimate[i-2]) / gr_complex(2.0, 0.0);
  }

  // with even number of carriers; last equalizer tap is wrong
  if(!(d_occupied_carriers & 1)) {
    d_hestimate[d_occupied_carriers-1] = d_hestimate[d_occupied_carriers-2];
  }

  // linklab, calculate channel gain
  std::stringstream ss;
  for(i = 0; i < d_occupied_carriers; i++) {
    ss << 1.0/(abs(d_hestimate[i]));
    ss << " ";
  }
  d_ch = ss.str();

  // linklab, print the estimated channel coefficient
  /*printf("channel estimation: \n");
  for(i = 0; i < d_occupied_carriers; i++) {
    printf("%f%+fi,",d_hestimate[i].real(), d_hestimate[i].imag());
  }*/
  
  // linklab, print the received symbol
  /*printf("received symbol: \n");
  for(i = 0; i < d_fft_length; i++) {
    printf("%f%+fi,",symbol[i].real(), symbol[i].imag());
  }*/

  if(VERBOSE) {
    fprintf(stderr, "Equalizer setting:\n");
    for(i = 0; i < d_occupied_carriers; i++) {
      gr_complex sym = coarse_freq_comp(d_coarse_freq,1)*symbol[i+zeros_on_left+d_coarse_freq];
      gr_complex output = sym * d_hestimate[i];
      fprintf(stderr, "sym: %+.4f + j%+.4f  ks: %+.4f + j%+.4f  eq: %+.4f + j%+.4f  ==>  %+.4f + j%+.4f\n", 
	      sym .real(), sym.imag(),
	      d_known_symbol_tmp[i].real(), d_known_symbol_tmp[i].imag(),
	      d_hestimate[i].real(), d_hestimate[i].imag(),
	      output.real(), output.imag());
    }
    fprintf(stderr, "\n");
  }
}

// linklab, estimate SINR
void
gr_ofdm_frame_acquisition::calculate_sinr(const gr_complex *symbol, int zeros_on_left)
{
    unsigned int i = 0;
    unsigned int carrier_num = 0;
    float overall_power = 0;
    float signal_noise_power = 0;
    float noise_power = 0;

    // caculate the overall power
    for (i=0; i< d_fft_length; i++){
        overall_power += pow(symbol[i+d_coarse_freq].real(), 2)+pow(symbol[i+d_coarse_freq].imag(), 2); //linklab
    }
 
    // caculate the signal power
    for (i=0; i<d_occupied_carriers-1; i+=2){
        if (d_known_symbol_tmp[i].real() != 0){
	    carrier_num += 2;
            signal_noise_power += pow(symbol[i+d_coarse_freq+zeros_on_left].real(), 2) +pow(symbol[i+d_coarse_freq+zeros_on_left].imag(), 2)
			   +pow(symbol[i+1+d_coarse_freq+zeros_on_left].real(), 2) +pow(symbol[i+1+d_coarse_freq+zeros_on_left].imag(), 2);
	}
    }

    // caculate the noise power
    noise_power = (overall_power - signal_noise_power);

    // caculate the sinr
    d_sinr = 10*log((signal_noise_power/noise_power)-1)/log(10);

    // printf("Frequency domain estimated SNR: %.1f dB\n", d_sinr);
}

int
gr_ofdm_frame_acquisition::general_work(int noutput_items,
					gr_vector_int &ninput_items,
					gr_vector_const_void_star &input_items,
					gr_vector_void_star &output_items)
{
  const gr_complex *symbol = (const gr_complex *)input_items[0];
  const char *signal_in = (const char *)input_items[1];

  gr_complex *out = (gr_complex *) output_items[0];
  char *signal_out = (char *) output_items[1];
  
  int unoccupied_carriers = d_fft_length - d_occupied_carriers;
  int zeros_on_left = (int)ceil(unoccupied_carriers/2.0);

  if(signal_in[0]) {
    d_phase_count = 1;
    correlate(symbol, zeros_on_left);
    calculate_equalizer(symbol, zeros_on_left);
    // linklab, calculate the sinr in frequency domain
    calculate_sinr(symbol, zeros_on_left);
    signal_out[0] = 1;
  }
  else {
    signal_out[0] = 0;
  } 

  for(unsigned int i = 0; i < d_occupied_carriers; i++) {
    out[i] = d_hestimate[i]*coarse_freq_comp(d_coarse_freq,d_phase_count)
      *symbol[i+zeros_on_left+d_coarse_freq];
  }
  
  d_phase_count++;
  if(d_phase_count == MAX_NUM_SYMBOLS) {
    d_phase_count = 1;
  }

  consume_each(1);
  return 1;
}
