/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
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
// This is a modification of gr_ofdm_frame_acquisition.cc from GNU Radio.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_ofdm_frame_acquisition.h>
#include <gr_io_signature.h>
#include <gr_expj.h>
#include <gr_math.h>
#include <cstdio>

#define VERBOSE 0
#define M_TWOPI (2*M_PI)
#define MAX_NUM_SYMBOLS 1000

foimimo_ofdm_frame_acquisition_sptr
foimimo_make_ofdm_frame_acquisition (unsigned int occupied_carriers, unsigned int fft_length,
				unsigned int cplen,
				const std::vector<gr_complex> &known_symbol,
				unsigned int max_fft_shift_len)
{
  return gnuradio::get_initial_sptr(new foimimo_ofdm_frame_acquisition (occupied_carriers, fft_length, cplen,
									known_symbol, max_fft_shift_len));
}

foimimo_ofdm_frame_acquisition::foimimo_ofdm_frame_acquisition (unsigned occupied_carriers, unsigned int fft_length,
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
    d_phase_count(0)
{
  d_symbol_phase_diff.resize(d_fft_length);
  d_known_phase_diff.resize(d_occupied_carriers);
  d_hestimate.resize(d_occupied_carriers);

  unsigned int i = 0, j = 0;

  std::fill(d_known_phase_diff.begin(), d_known_phase_diff.end(), 0);
  for(i = 0; i < d_known_symbol.size()-2; i+=2) {
    d_known_phase_diff[i] = norm(d_known_symbol[i] - d_known_symbol[i+2]);
  }
  
  d_phase_lut = new gr_complex[(2*d_freq_shift_len+1) * MAX_NUM_SYMBOLS];
  for(i = 0; i <= 2*d_freq_shift_len; i++) {
    for(j = 0; j < MAX_NUM_SYMBOLS; j++) {
      d_phase_lut[j + i*MAX_NUM_SYMBOLS] =  gr_expj(-M_TWOPI*d_cplen/d_fft_length*(i-d_freq_shift_len)*j);
    }
  }
  d_first_preamble = false;
}

foimimo_ofdm_frame_acquisition::~foimimo_ofdm_frame_acquisition(void)
{
  delete [] d_phase_lut;
}

void
foimimo_ofdm_frame_acquisition::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  unsigned ninputs = ninput_items_required.size ();
  for (unsigned i = 0; i < ninputs; i++)
    ninput_items_required[i] = 1;
}

gr_complex
foimimo_ofdm_frame_acquisition::coarse_freq_comp(int freq_delta, int symbol_count)
{

  return gr_expj(-M_TWOPI*freq_delta*d_cplen/d_fft_length*symbol_count);
}

void
foimimo_ofdm_frame_acquisition::correlate(const gr_complex *symbol, int zeros_on_left)
{
  unsigned int i,j;
  
  std::fill(d_symbol_phase_diff.begin(), d_symbol_phase_diff.end(), 0);
  for(i = 0; i < d_fft_length-2; i++) {
    d_symbol_phase_diff[i] = norm(symbol[i] - symbol[i+2]);
  }

  // sweep through all possible/allowed frequency offsets and select the best
  int index = 0;
  float max = 0, sum=0;
  for(i =  zeros_on_left - d_freq_shift_len; i < zeros_on_left + d_freq_shift_len; i++) {
    sum = 0;
    for(j = 0; j < d_occupied_carriers; j++) {
      sum += (d_known_phase_diff[j] * d_symbol_phase_diff[i+j]);
    }
    if(sum > max) {
      max = sum;
      index = i;
    }
  }
  
  // set the coarse frequency offset relative to the edge of the occupied tones
  d_coarse_freq = index - zeros_on_left;
}

void
foimimo_ofdm_frame_acquisition::calculate_equalizer(const gr_complex *symbol, int zeros_on_left)
{
  unsigned int i=0;

  for(i = 0; i < d_occupied_carriers; i++) {
    d_hestimate[i] = d_known_symbol[i] / 
      (coarse_freq_comp(d_coarse_freq,1)*(symbol[i+zeros_on_left+d_coarse_freq]));
  }

  if(VERBOSE) {
    fprintf(stderr, "Equalizer setting:\n");
    for(i = 0; i < d_occupied_carriers; i++) {
      gr_complex sym = coarse_freq_comp(d_coarse_freq,1)*symbol[i+zeros_on_left+d_coarse_freq];
      gr_complex output = sym * d_hestimate[i];
      fprintf(stderr, "sym: %+.4f + j%+.4f  ks: %+.4f + j%+.4f  eq: %+.4f + j%+.4f  ==>  %+.4f + j%+.4f\n", 
	      sym .real(), sym.imag(),
	      d_known_symbol[i].real(), d_known_symbol[i].imag(),
	      d_hestimate[i].real(), d_hestimate[i].imag(),
	      output.real(), output.imag());
    }
    fprintf(stderr, "\n");
  }
}

int
foimimo_ofdm_frame_acquisition::general_work(int noutput_items,
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
    d_first_preamble = true;
    signal_out[0] = 0;
  }
  else if(d_first_preamble){
    d_first_preamble = false;
    d_phase_count = 1;
    correlate(symbol, zeros_on_left);
    calculate_equalizer(symbol, zeros_on_left);
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
