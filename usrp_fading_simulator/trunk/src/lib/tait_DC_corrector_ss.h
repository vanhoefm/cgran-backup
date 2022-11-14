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
#ifndef INCLUDED_DC_CORRECTOR_SS_H
#define INCLUDED_DC_CORRECTOR_SS_H

#include <gr_sync_block.h>

extern "C" 
{
	void spas_dcCorrector_ss(int16_t *input, int16_t *output, const uint16_t numberSamples, 
			      const int16_t dc_offset_remove_const);
}

class tait_DC_corrector_ss;

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
typedef boost::shared_ptr<tait_DC_corrector_ss> tait_DC_corrector_ss_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of tait_DC_corrector_ss.
 *
 * To avoid accidental use of raw pointers, tait_DC_corrector_ss's
 * constructor is private.  tait_make_DC_corrector_ss is the public
 * interface for creating new instances.
 */
//(dc_offset_remove_const would idealy be a int16_t type but python complained).
tait_DC_corrector_ss_sptr tait_make_DC_corrector_ss (const int dc_offset_remove_const);

/*!
 * \brief Removes any DC offset.
 * \ingroup block
 *
 * The dc_offset_remove_const dictates how quickly the DC removal adapts.
 * The higher the value the slower the adaption but also the less the attenuation.
 * dc_offset_remove_const should be greater than 0 and less than 32768.
 */
class tait_DC_corrector_ss : public gr_sync_block
{
private:
  // The friend declaration allows tait_make_DC_corrector_ss to
  // access the private constructor.

	friend tait_DC_corrector_ss_sptr tait_make_DC_corrector_ss (const int dc_offset_remove_const);

  tait_DC_corrector_ss (const int16_t dc_offset_remove_const);  	// private constructor
  
  // Related to the rate of DC removal adpation.
  const int16_t offset_const;

 public:
  ~tait_DC_corrector_ss ();	// public destructor

  // Where all the action really happens

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_DC_CORRECTOR_H */
