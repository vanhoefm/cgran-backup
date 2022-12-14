/* -*- c++ -*- */
/*
 * Copyright 2011 Free Software Foundation, Inc.
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

#include <grgpu_d2h_c_cuda.h>
#include <gr_io_signature.h>
#include <stdio.h>
/*
 * Create a new instance of grgpu_d2h_c_cuda and return
 * a boost shared_ptr.  This is effectively the public constructor.
 */
grgpu_d2h_c_cuda_sptr 
grgpu_make_d2h_c_cuda ()
{
  return grgpu_d2h_c_cuda_sptr (new grgpu_d2h_c_cuda ());
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
grgpu_d2h_c_cuda::grgpu_d2h_c_cuda ()
  : gr_block ("d2h_c_cuda",
	      gr_make_io_signature (MIN_IN, MAX_IN, sizeof (unsigned long)),
	      gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof (float)*2))
{
  d_verbose = 0;
  this->set_output_multiple(1024);
}

/*
 * Our virtual destructor.
 */
grgpu_d2h_c_cuda::~grgpu_d2h_c_cuda ()
{
  // nothing else required in this example
}


/*
 * The C hook into our cuda kernel call
 */
void grgpu_d2h_c_cuda_work_device(int noutput_items, const unsigned long* input_items, float* output_items);


int 
grgpu_d2h_c_cuda::general_work (int noutput_items,
			       gr_vector_int &ninput_items,
			       gr_vector_const_void_star &input_items,
			       gr_vector_void_star &output_items)
{


  int i;
  if(d_verbose)
    printf("Executing Work function for D2H. %d\n", noutput_items);
  const unsigned long *in = (const unsigned long *) input_items[0];
  float *out = (float *) output_items[0];


   
   grgpu_d2h_c_cuda_work_device(noutput_items, in, out);
   if(d_verbose)
     printf("D2H, I/O: %ld, %f , noutput:%d.\n", in[0], out[0], noutput_items);

   consume_each (noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
