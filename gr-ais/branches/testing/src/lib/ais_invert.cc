//ais_invert.cc
/* -*- c++ -*- */
/*
 * Copyright 2004 Free Software Foundation, Inc.
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

/*
 * config.h is generated by configure.  It contains the results
 * of probing for features, options etc.  It should be the first
 * file included in your .cc file.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ais_invert.h>
#include <gr_io_signature.h>

/*
 * Create a new instance of ais_invert and return
 * a boost shared_ptr.  This is effectively the public constructor.
 */
ais_invert_sptr ais_make_invert()
{
  return ais_invert_sptr (new ais_invert ());
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
static const int MIN_IN = 1;    // mininum number of input streams
static const int MAX_IN = 1;    // maximum number of input streams
static const int MIN_OUT = 1;   // minimum number of output streams
static const int MAX_OUT = 1;   // maximum number of output streams

/*
 * The private constructor
 */
ais_invert::ais_invert ()
  : gr_sync_block ("invert",
                   gr_make_io_signature (MIN_IN, MAX_IN, sizeof (char)),
                   gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof (char)))
{
  // nothing else required in this example
}

/*
 * Our virtual destructor.
 */
ais_invert::~ais_invert ()
{
  // nothing else required in this example
}

int 
ais_invert::work (int noutput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items)
{
  const char *in = (const char *) input_items[0];
  char *out = (char *) output_items[0];

  for (int i = 0; i < noutput_items; i++){
          if (in[i] == 1) out[i] = 0;
          else if (in[i] == 0) out[i] = 1;
//          else printf("Non-binary input to invert()\n"); //should probably flag an error somewhere
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

