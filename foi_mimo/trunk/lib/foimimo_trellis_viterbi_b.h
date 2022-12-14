/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
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
// This is a modification of trellis_viterbi_b.h from GNU Radio.

#ifndef INCLUDED_foimimo_TRELLIS_VITERBI_B_H
#define INCLUDED_foimimo_TRELLIS_VITERBI_B_H

#include "fsm.h"
#include <gr_block.h>

class foimimo_trellis_viterbi_b;
typedef boost::shared_ptr<foimimo_trellis_viterbi_b> foimimo_trellis_viterbi_b_sptr;

foimimo_trellis_viterbi_b_sptr
foimimo_make_trellis_viterbi_b (const fsm &FSM, int K, int S0, int SK);

/*!
 *  \ingroup coding_blk
 */
class foimimo_trellis_viterbi_b : public gr_block
{
  fsm d_FSM;
  int d_K;
  int d_S0;
  int d_SK;
  int d_default_state;

  friend foimimo_trellis_viterbi_b_sptr foimimo_make_trellis_viterbi_b (
    const fsm &FSM, int K, int S0, int SK);


  foimimo_trellis_viterbi_b (const fsm &FSM,int K, int S0,int SK);


public:
  fsm FSM () const { return d_FSM; }
  int K () const { return d_K; }
  int S0 () const { return d_S0; }
  int SK () const { return d_SK; }
  void set_K(int K);

  void forecast (int noutput_items, gr_vector_int &ninput_items_required);

  int general_work (int noutput_items,
                    gr_vector_int &ninput_items,
                    gr_vector_const_void_star &input_items,
                    gr_vector_void_star &output_items);
};

#endif
