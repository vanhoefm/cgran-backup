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

#ifndef INCLUDED_GR_NRZI_TO_NRZ_BB_H
#define INCLUDED_GR_NRZI_TO_NRZ_BB_H

#include <gr_sync_block.h>

class gr_nrzi_to_nrz_bb;
typedef boost::shared_ptr<gr_nrzi_to_nrz_bb> gr_nrzi_to_nrz_bb_sptr;

gr_nrzi_to_nrz_bb_sptr gr_make_nrzi_to_nrz_bb (bool preload);

/*!
 * \brief Convert NRZI to NRZ. One bit per byte.
 * \ingroup block
 *
 */
class gr_nrzi_to_nrz_bb : public gr_sync_block
{
  friend gr_nrzi_to_nrz_bb_sptr gr_make_nrzi_to_nrz_bb (bool preload);
  gr_nrzi_to_nrz_bb (bool preload);

  unsigned char d_prev_nrzi_bit;

 public:
  int work (int                       noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star       &output_items);
};

#endif
