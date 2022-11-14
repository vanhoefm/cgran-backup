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
#ifndef INCLUDED_tait_SQUARE2_FF_H
#define INCLUDED_tait_SQUARE2_FF_H

#include <gr_sync_block.h>
#include <gr_io_signature.h>
				
extern "C" 
{
	extern void lib_Biquad4(int16_t *input, int16_t *coeffs, int16_t *output,\
		int16_t *delays, uint16_t nBiquads, uint16_t nSamples);
}

class tait_biquad4_ss;

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
typedef boost::shared_ptr<tait_biquad4_ss> tait_biquad4_ss_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of tait_biquad4_ss.
 *
 * To avoid accidental use of raw pointers, tait_biquad4_ss's
 * constructor is private.  tait_make_biquad4_ss is the public
 * interface for creating new instances.
 */
tait_biquad4_ss_sptr tait_make_biquad4_ss ();

/*!
 * \brief Biquad fillter.
 * \ingroup block
 *
 * This uses the preferred technique: subclassing gr_sync_block.
 */
class tait_biquad4_ss : public gr_sync_block
{
private:
  // The friend declaration allows tait_make_biquad4_ss to
  // access the private constructor.

  friend tait_biquad4_ss_sptr tait_make_biquad4_ss ();

  tait_biquad4_ss ();  	// private constructor

 public:
  ~tait_biquad4_ss ();	// public destructor

  // Where all the action really happens

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_tait_SQUARE2_FF_H */
