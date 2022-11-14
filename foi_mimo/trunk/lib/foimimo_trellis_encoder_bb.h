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
// This is a modification of trellis_encoder_bb.h from GNU Radio.

#ifndef INCLUDED_foimimo_TRELLIS_ENCODER_BB_H
#define INCLUDED_foimimo_TRELLIS_ENCODER_BB_H

#include <fsm.h>
#include <gr_sync_block.h>

class foimimo_trellis_encoder_bb;
typedef boost::shared_ptr<foimimo_trellis_encoder_bb> foimimo_trellis_encoder_bb_sptr;

foimimo_trellis_encoder_bb_sptr foimimo_make_trellis_encoder_bb (const fsm &FSM, int ST);

/*!
 * \brief Convolutional encoder. This block gets an entire OFDM packet payload
 *        at input0 and the header for the packet at input1. The heder is first
 *        outputed uncoded flowed by the coded payload. A new packet flag is raised
 *        for each packet so the the next block knows when a new packet begins.
 *
 * \param FSM Encoder state-machine definition
 * \param ST initial state
 * \param byte_per_packet number of bytes in one OFDM packet.
 *
 */
class foimimo_trellis_encoder_bb : public gr_sync_block
{
private:
  friend foimimo_trellis_encoder_bb_sptr foimimo_make_trellis_encoder_bb (const fsm &FSM, int ST);
  fsm d_FSM;
  int d_ST;
  foimimo_trellis_encoder_bb (const fsm &FSM, int ST);

  static const int HEADER_SIZE = 4;

  unsigned char d_input_mask;

  enum state_t {HEADER, PAYLOAD};
  state_t d_state;

  unsigned int d_packet_len;
  unsigned int d_byte_cnt;

  void enter_header();
  void enter_payload(const unsigned char* header);

public:
  fsm FSM () const { return d_FSM; }
  int ST () const { return d_ST; };

  int work (int noutput_items,
      gr_vector_const_void_star &input_items,
      gr_vector_void_star &output_items);

};

#endif
