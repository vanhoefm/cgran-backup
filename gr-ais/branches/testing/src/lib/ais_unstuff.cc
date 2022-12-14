//ais_unstuff.cc
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

#include <ais_unstuff.h>
#include <gr_io_signature.h>

/*
 * Create a new instance of ais_unstuff and return
 * a boost shared_ptr.  This is effectively the public constructor.
 */
ais_unstuff_sptr ais_make_unstuff()
{
  return ais_unstuff_sptr (new ais_unstuff ());
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
ais_unstuff::ais_unstuff ()
  : gr_block ("unstuff",
                   gr_make_io_signature (MIN_IN, MAX_IN, sizeof (char)),
                   gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof (char)))
{
  set_relative_rate((double)(258.0/258.0));
  d_consecutive = 0;
  //printf("Calling constructor\n");
}

/*
 * Our virtual destructor.
 */
ais_unstuff::~ais_unstuff ()
{
  // nothing else required in this example
}

void ais_unstuff::forecast (int noutput_items,
	       gr_vector_int &ninput_items_required) //estimate number of input samples required for noutput_items samples
{
	int size = noutput_items + 2*(noutput_items / 256); //on average

	ninput_items_required[0] = size;
}


//so is this is gonna unstuff the start and stop codes? nooooo, because those have a following '1'.
int 
ais_unstuff::general_work (int noutput_items,
		                gr_vector_int &ninput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items)
{
  const char *in = (const char *) input_items[0];
  char *out = (char *) output_items[0];

  int j = 0;
  int i = 0;
  //printf("Entering ais_unstuff::general_work with %i requested output items and %i offered input items\n", noutput_items, ninput_items[0]);

  while(i < noutput_items){
	//printf("i is %i\n", i);
	//printf("\td_consecutive is %i\n", d_consecutive);
	//printf("\tinput is %i\n", in[i]);
	if(in[i] & 0x01) {//if bit 0 is set (the data bit)
		d_consecutive++;
	} else {
		if(d_consecutive == 5) {
			//printf("Tossing!\n");
			i++;
		}
		d_consecutive = 0;
	}
	if(i >= noutput_items) break;
	out[j++] = in[i++];
  }

	//printf("Total in: %i, total out: %i\n", i, j);
  consume_each(i); //tell gnuradio how many input items we used
//  ninput_items[0] = i;
  // Tell runtime system how many output items we produced.
  return j;
}

