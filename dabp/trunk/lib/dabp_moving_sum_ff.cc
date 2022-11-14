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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dabp_moving_sum_ff.h>
#include <gr_io_signature.h>
#include <cmath>

dabp_moving_sum_ff_sptr 
dabp_make_moving_sum_ff (int length, double alpha)
{
  return dabp_moving_sum_ff_sptr (new dabp_moving_sum_ff (length, alpha));
}

static const int MIN_IN = 1;  // mininum number of input streams
static const int MAX_IN = 1;  // maximum number of input streams
static const int MIN_OUT = 1; // minimum number of output streams
static const int MAX_OUT = 1; // maximum number of output streams

/*
 * The private constructor
 */
dabp_moving_sum_ff::dabp_moving_sum_ff (int length, double alpha)
        : gr_sync_block ("moving_sum_ff",
                gr_make_io_signature (MIN_IN, MAX_IN, sizeof (float)),
                gr_make_io_signature (MIN_OUT, MAX_OUT, sizeof (float))),
        d_sum(0), d_length(length), d_alpha(alpha)
{
    d_alphapwr=std::pow(alpha,(double)length);
    assert(length>=0);
    set_history(length+1);
}

dabp_moving_sum_ff::~dabp_moving_sum_ff ()
{
}

int dabp_moving_sum_ff::work (int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
{
    const float *in = (const float *) input_items[0];
    float *out = (float *) output_items[0];

    for (int i=0; i < noutput_items; i++) {
        d_sum=d_sum*d_alpha+(double)in[i+d_length]-(double)in[i]*d_alphapwr;
        out[i] = (float)d_sum;
    }

    return noutput_items;
}

