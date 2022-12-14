/* -*- c++ -*- */
/*
 * Copyright 2008 Free Software Foundation, Inc.
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

#ifndef INCLUDED_GR_DERANDOMIZE_BB_H
#define INCLUDED_GR_DERANDOMIZE_BB_H

#include <gr_sync_block.h>

class gr_derandomize_bb;
typedef boost::shared_ptr<gr_derandomize_bb> gr_derandomize_bb_sptr;

gr_derandomize_bb_sptr gr_make_derandomize_bb (unsigned int tap_mask, unsigned int preload);

/*!
 * \brief Derandomize a bitstream. One bit per byte, in the low-order bit. 
 * Uses pseudorandom sequences up to (2^32)-1 bits long.
 * \ingroup block
 *
 */
class gr_derandomize_bb : public gr_sync_block
{
  friend gr_derandomize_bb_sptr gr_make_derandomize_bb (unsigned int tap_mask, unsigned int preload);
  gr_derandomize_bb (unsigned int tap_mask, unsigned int preload);

  unsigned int d_shift_register;
  int          d_taps[32];
  int          d_tap_count;

 public:
  int work (int                       noutput_items,
	        gr_vector_const_void_star &input_items,
	        gr_vector_void_star       &output_items);
};

#endif
