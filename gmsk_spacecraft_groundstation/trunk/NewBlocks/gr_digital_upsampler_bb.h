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
#ifndef INCLUDED_GR_DIGITAL_UPSAMPLER_BB_H
#define INCLUDED_GR_DIGITAL_UPSAMPLER_BB_H

#include <gr_sync_block.h>

class gr_digital_upsampler_bb;

/*
 * We use boost::shared_ptr's instead of raw pointers for all access
 * to gr_blocks (and many other data structures).  The shared_ptr gets
 * us transparent reference counting, which greatly simplifies storage
 * management issues.  This is especially helpful in our hybrid
 * C++ / Python system.
 *
 * See http://www.boost.org/libs/smart_ptr/smart_ptr.htm
 *
 * As a convention, the _sptr suffix indicates a boost::shared_ptr
 */
typedef boost::shared_ptr<gr_digital_upsampler_bb> gr_digital_upsampler_bb_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of gr_digital_upsampler_bb.
 *
 * To avoid accidental use of raw pointers, gr_digital_upsampler_bb's
 * constructor is private.  gr_make_digital_upsampler_bb is the public
 * interface for creating new instances.
 */
gr_digital_upsampler_bb_sptr gr_make_digital_upsampler_bb (unsigned int input_rate, 
                                                           unsigned int output_rate);

/*!
 * \brief Upsamples a digital stream of bits carried in the lsb of a byte.
 * \ingroup block
 *
 */
class gr_digital_upsampler_bb : public gr_block
{
private:
  // The friend declaration allows gr_make_digital_upsampler_bb to
  // access the private constructor.

  friend gr_digital_upsampler_bb_sptr gr_make_digital_upsampler_bb (unsigned int input_rate,
                                                                    unsigned int output_rate);

  unsigned int d_input_rate;
  unsigned int d_output_rate;
  const float  d_input_sample_time;
  const float  d_output_sample_time;
  float        d_remainder_time;

  gr_digital_upsampler_bb (unsigned int input_rate, 
                           unsigned int output_rate);  	// private constructor

 public:
  ~gr_digital_upsampler_bb ();	// public destructor

  void forecast ( int             noutput_items,
                  gr_vector_int   &ninput_items_required);

  // Where all the action really happens

  int general_work (int noutput_items,
	    gr_vector_int &ninput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_GR_DIGITAL_UPSAMPLER_BB_H */
