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

/**
 * This is a modification of the gr_multichanneladd_cc from GNURadio.
 *
 * Modified by: Thomas Schmid <thomas.schmid@ucla.edu>
 */

#ifndef INCLUDED_UCLA_MULTICHANNELADD_CC_H
#define INCLUDED_UCLA_MULTICHANNELADD_CC_H

#include <gr_sync_interpolator.h>

class ucla_multichanneladd_cc;
typedef boost::shared_ptr<ucla_multichanneladd_cc> ucla_multichanneladd_cc_sptr;

ucla_multichanneladd_cc_sptr gr_make_multichanneladd_cc (size_t itemsize);

/*!
 * \brief multichanneladd_cc N inputs to a single output
 * \ingroup block
 */
class ucla_multichanneladd_cc : public gr_block
{
  friend ucla_multichanneladd_cc_sptr ucla_make_multichanneladd_cc (size_t itemsize);

  size_t	d_itemsize;

  ucla_multichanneladd_cc (size_t itemsize);

public:
  ~ucla_multichanneladd_cc ();

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items);

  // gr_sync_block overrides these to assist work
  void forecast (int noutput_items, gr_vector_int &ninput_items_required);
  int  general_work (int noutput_items,
		     gr_vector_int &ninput_items,
		     gr_vector_const_void_star &input_items,
		     gr_vector_void_star &output_items);

};

#endif /* INCLUDED_UCLA_MULTICHANNELADD_CC_H */
