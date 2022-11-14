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

#include <foimimo_chunk_2_byte.h>
#include <gr_io_signature.h>
#include <stdio.h>

foimimo_chunk_2_byte_sptr
foimimo_make_chunk_2_byte(unsigned int chunk_size){
  return foimimo_chunk_2_byte_sptr (new foimimo_chunk_2_byte(chunk_size));
}

foimimo_chunk_2_byte::foimimo_chunk_2_byte(unsigned int chunk_size)
  :gr_block ("chunk_2_byte",
      gr_make_io_signature2 (2, 2, sizeof (char), sizeof(char)),
      gr_make_io_signature2 (2, 2, sizeof (char), sizeof(char))),
  d_chunk_size(chunk_size),
  d_residbit(0),
  d_residbit_cnt(0),
  d_residbit_flag(0),
  d_residbit_flag_cnt(0)
{
  set_relative_rate(d_chunk_size/8.0);
}

void
foimimo_chunk_2_byte::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  int input_required = ceil(noutput_items*8.0/d_chunk_size);

  ninput_items_required[0] = input_required;
  ninput_items_required[1] = input_required;
}
int
foimimo_chunk_2_byte::general_work (int noutput_items,
                  gr_vector_int &ninput_items,
                  gr_vector_const_void_star &input_items,
                  gr_vector_void_star &output_items)
{
  const unsigned char *in = (const unsigned char*) input_items[0];
  const unsigned char *in_new_pkt = (const unsigned char*) input_items[1];
  unsigned char *out = (unsigned char*) output_items[0];
  unsigned char *out_new_pkt = (unsigned char*) output_items[1];

  assert(ninput_items[0]==ninput_items[1]);

  int nr_in_chunks = std::min(ninput_items[0],ninput_items[1]);

  unsigned int i = 0;
  unsigned int out_cnt = 0;

  while(i < nr_in_chunks && out_cnt < noutput_items){
    d_residbit = (d_residbit << d_chunk_size) | (in[i] & ((1 << d_chunk_size)-1));
    d_residbit_cnt += d_chunk_size;


    if (d_residbit_cnt > 7){
      out_new_pkt[out_cnt] = 0;
      out[out_cnt++] = (d_residbit >> (d_residbit_cnt -8)) & 0xff;
      d_residbit_cnt -= 8;
    }
    if (in_new_pkt[i] == 1){
      out_new_pkt[out_cnt-1] = 1;
      d_residbit_cnt = 0;
      d_residbit = 0;
    }

    i++;
  }
 
  assert(out_cnt <= noutput_items);
  consume_each(i);
  return out_cnt;
}
