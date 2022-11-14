/* -*- c++ -*- */
/*
 * Copyright 2004 Free Software Foundation, Inc.
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

/*
 * config.h is generated by configure.  It contains the results
 * of probing for features, options etc.  It should be the first
 * file included in your .cc file.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gr_digital_upsampler_bb.h>
#include <gr_io_signature.h>

/*
 * Create a new instance of gr_digital_upsampler_bb and return
 * a boost shared_ptr.  This is effectively the public constructor.
 */
gr_digital_upsampler_bb_sptr 
gr_make_digital_upsampler_bb (unsigned int input_rate, unsigned int output_rate)
{
  return gr_digital_upsampler_bb_sptr (new gr_digital_upsampler_bb (input_rate, output_rate));
}

/*
 * Specify constraints on number of input and output streams.
 * This info is used to construct the input and output signatures
 * (2nd & 3rd args to gr_block's constructor).  The input and
 * output signatures are used by the runtime system to
 * check that a valid number and type of inputs and outputs
 * are connected to this block.  In this case, we accept
 * only 1 input and 1 output.
 */
static const int MIN_IN = 1;	// mininum number of input streams
static const int MAX_IN = 1;	// maximum number of input streams
static const int MIN_OUT = 1;	// minimum number of output streams
static const int MAX_OUT = 1;	// maximum number of output streams

/*
 * The private constructor
 */
gr_digital_upsampler_bb::gr_digital_upsampler_bb (unsigned int input_rate, 
                                                  unsigned int output_rate)
  : gr_block ("digital_upsampler_bb",
              gr_make_io_signature (MIN_IN, MAX_IN, sizeof (unsigned char)),
	      gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof (unsigned char))),
    d_input_rate(input_rate),
    d_output_rate(output_rate),
    d_input_sample_time(1.0/input_rate),
    d_output_sample_time(1.0/output_rate),
    d_remainder_time(0.0)
{
    set_relative_rate(d_output_rate/d_input_rate);
}

/*
 * Our virtual destructor.
 */
gr_digital_upsampler_bb::~gr_digital_upsampler_bb ()
{
  // nothing else required
}


void gr_digital_upsampler_bb::forecast ( int             noutput_items,
                                         gr_vector_int   &ninput_items_required)
{
    int n = noutput_items * d_input_rate / d_output_rate;
    ninput_items_required[0] = (n==0 ? 1 : n);
}


int 
gr_digital_upsampler_bb::general_work (int                       noutput_items,
                                       gr_vector_int             &ninput_items,
                                       gr_vector_const_void_star &input_items,
                                       gr_vector_void_star       &output_items)
{
  const unsigned char *in  = (const unsigned char *) input_items[0];
  unsigned char       *out = (unsigned char *)       output_items[0];
  float                elapsed_time;
  int                  n_in_items = ninput_items[0];
  int                  i; // input item counter
  int                  o; // output item counter

  elapsed_time = d_remainder_time;  // get any leftover time from last call

  // Loop thru all output items 'till we run out of output or input
  i = 0;
  for (o = 0; (o < noutput_items) && (i < n_in_items); o++)
  {    
    elapsed_time += d_output_sample_time;  // one output sample time elapsed
    if(elapsed_time >= d_input_sample_time) // see if we've started a new bit
    {
      elapsed_time -= d_input_sample_time;  // adjust by one input bit time
      i++;                                  // move to next input bit
    }
    out[o] = in[i];                        // copy input bit value to output
  }

  d_remainder_time = elapsed_time;  // save any remainder time for next call

  // Tell runtime system how many input items we consumed on
  // each input stream.
    consume_each (i);

  // Tell runtime system how many output items we produced.
  return o;
}