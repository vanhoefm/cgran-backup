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

#ifndef FOIMIMO_CHUNK_2_BYTE_H_
#define FOIMIMO_CHUNK_2_BYTE_H_

#include <gr_block.h>

class foimimo_chunk_2_byte;
typedef boost::shared_ptr<foimimo_chunk_2_byte> foimimo_chunk_2_byte_sptr;

foimimo_chunk_2_byte_sptr foimimo_make_chunk_2_byte(unsigned int chunk_size);
/*!
 * \brief Puts together a tuple (chunk) of bits into a byte. The tuples are shifted
 *        upwards to MSB.
 *
 * \param chunk_size the number of bits in one chunk.
 *
 */

class foimimo_chunk_2_byte : public gr_block
{
  friend foimimo_chunk_2_byte_sptr foimimo_make_chunk_2_byte(unsigned int chunk_size);

  foimimo_chunk_2_byte(unsigned int chunk_size);

  unsigned int d_chunk_size;
  unsigned int d_residbit;
  unsigned int d_residbit_cnt;
  unsigned int d_residbit_flag;
  unsigned int d_residbit_flag_cnt;

public:
  void forecast (int noutput_items,
                 gr_vector_int &ninput_items_required);
  int general_work (int noutput_items,
                    gr_vector_int &ninput_items,
                    gr_vector_const_void_star &input_items,
                    gr_vector_void_star &output_items);
};


#endif /* FOIMIMO_CHUNK_2_BYTE_H_ */
