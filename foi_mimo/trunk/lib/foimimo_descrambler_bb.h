/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2008 Free Software Foundation, Inc.
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
// This is a modification of gr_descrambler_bb.h from GNU Radio.

#ifndef INCLUDED_FOI_DESCRAMBLER
#define INCLUDED_FOI_DESCRAMBLER

#include <gr_sync_block.h>

class foimimo_descrambler_bb;
typedef boost::shared_ptr<foimimo_descrambler_bb> foimimo_descrambler_bb_sptr;

foimimo_descrambler_bb_sptr
foimimo_make_descrambler_bb();

/*!
 * deScramble an input stream using an LFSR. When the new packet flag is
 * signaled the shift register is rest to its initial seed.
 *
 * \param mask     Polynomial mask for LFSR
 * \param seed     Initial shift register contents
 * \param len      Shift register length
 *
 * \ingroup coding_blk
 */

class foimimo_descrambler_bb: public gr_sync_block
{
  friend foimimo_descrambler_bb_sptr
  foimimo_make_descrambler_bb();

  foimimo_descrambler_bb();

  unsigned int d_random_mask_index;


static const int HEADER_SIZE=4; //Header is 4 byte

public:
  int work(int noutput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};


#endif
