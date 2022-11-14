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
// This is a modification of gr_scrambler_bb.h from GNU Radio.

#ifndef INCLUDED_FOI_SCRAMBLER
#define INCLUDED_FOI_SCRAMBLER

#include <gr_sync_block.h>
#include "gri_lfsr.h"

class foimimo_scrambler_bb;
typedef boost::shared_ptr<foimimo_scrambler_bb> foimimo_scrambler_bb_sptr;

foimimo_scrambler_bb_sptr
foimimo_make_scrambler_bb(unsigned int byte_per_packet);

class foimimo_scrambler_bb: public gr_sync_block
{
  friend foimimo_scrambler_bb_sptr
  foimimo_make_scrambler_bb(unsigned int byte_per_packet);

  foimimo_scrambler_bb(unsigned int byte_per_packet);


  enum state_t {HEADER, PAYLOAD};
  state_t d_state;

  unsigned int d_byte_per_packet;
  static const int HEADER_SIZE = 4; //Header is 4 byte

  unsigned int d_packet_len;
  unsigned int d_byte_cnt;

  int d_mask_index;

  void enter_header();
  void enter_payload(const unsigned char* header);


public:
  int work(int noutput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};


#endif
