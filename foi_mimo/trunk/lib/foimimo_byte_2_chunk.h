/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 * 
 * This file is part of FOI-MIMO
 * 
 * FOI-MIMO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * FOI-MIMO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FOI-MIMO; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef FOIMIMO_BYTE_2_CHUNK_H_
#define FOIMIMO_BYTE_2_CHUNK_H_

#include <gr_block.h>

class foimimo_byte_2_chunk;
typedef boost::shared_ptr<foimimo_byte_2_chunk> foimimo_byte_2_chunk_sptr;

foimimo_byte_2_chunk_sptr foimimo_make_byte_2_chunk(unsigned int chunk_size);
/*!
 * \brief Puts together a tuple (chunk) of bits into a byte. The tuples are shifted
 *        upwards to MSB.
 *
 * \param chunk_size the number of bits in one chunk.
 *
 */

class foimimo_byte_2_chunk : public gr_block
{
  friend foimimo_byte_2_chunk_sptr foimimo_make_byte_2_chunk(unsigned int chunk_size);

  foimimo_byte_2_chunk(unsigned int chunk_size);

  static const int HEADER_SIZE = 4; //Header is 4 byte

  enum state_t {HEADER, PAYLOAD};
  state_t d_state;

  unsigned int d_input_mask;

  unsigned int d_chunk_size;
  unsigned int d_residbit;
  int d_residbit_cnt;

  unsigned int d_packet_len;
  int d_byte_cnt;

  void enter_header();
  void enter_payload(const unsigned char* header);

public:
  void forecast (int noutput_items,
                 gr_vector_int &ninput_items_required);
  int general_work (int noutput_items,
                    gr_vector_int &ninput_items,
                    gr_vector_const_void_star &input_items,
                    gr_vector_void_star &output_items);
};


#endif /* FOIMIMO_byte_2_chunk_H_ */
