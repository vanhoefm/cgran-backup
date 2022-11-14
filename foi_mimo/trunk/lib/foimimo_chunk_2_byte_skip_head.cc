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

#include <foimimo_chunk_2_byte_skip_head.h>
#include <gr_io_signature.h>
#include <stdio.h>

#define VERBOSE 0

void
foimimo_chunk_2_byte_skip_head::enter_header(){
  d_byte_cnt = 0;
  d_residbit = 0;
  d_residbit_cnt = 0;

  d_state = HEADER;
}

void
foimimo_chunk_2_byte_skip_head::enter_payload(const unsigned char* header){
  d_packet_len = (unsigned int) ((header[0] & 0xff) << 8) | (header[1] & 0xff);
  d_packet_len = ceil(d_packet_len*8.0/d_raw_bits) + HEADER_SIZE;
    if (VERBOSE)
      printf("chunk2byte@payload with packet_len:%i\n", d_packet_len);
  d_state = PAYLOAD;
}
foimimo_chunk_2_byte_skip_head_sptr
foimimo_make_chunk_2_byte_skip_head(unsigned int chunk_size, unsigned int raw_bits){
  return foimimo_chunk_2_byte_skip_head_sptr (new foimimo_chunk_2_byte_skip_head(chunk_size, raw_bits));
}

foimimo_chunk_2_byte_skip_head::foimimo_chunk_2_byte_skip_head(unsigned int chunk_size, unsigned int raw_bits)
  :gr_block ("chunk_2_byte",
      gr_make_io_signature2(2,2,sizeof(char),sizeof(char)),
      gr_make_io_signature2(2,2,sizeof(char),sizeof(char))),
  d_chunk_size(chunk_size),
  d_raw_bits(raw_bits),
  d_residbit(0),
  d_residbit_cnt(0)
{
  d_input_mask = (1 << d_chunk_size)-1;
  set_output_multiple(HEADER_SIZE);
  set_relative_rate(d_chunk_size/8.0);
  enter_header();
}

void
foimimo_chunk_2_byte_skip_head::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  int input_required = ceil(noutput_items*8.0/d_chunk_size);
  for(int i=0; i<ninput_items_required.size();i++){
    ninput_items_required[i] = input_required;
  }
}
int
foimimo_chunk_2_byte_skip_head::general_work (int noutput_items,
                  gr_vector_int &ninput_items,
                  gr_vector_const_void_star &input_items,
                  gr_vector_void_star &output_items)
{
  const unsigned char *in = (const unsigned char*) input_items[0];
  unsigned char *out = (unsigned char*) output_items[0];
  const unsigned char *in_new_pkt = (const unsigned char *) input_items[1];
  unsigned char *out_new_pkt = (unsigned char*) output_items[1];

  int nr_in_chunks = std::min(ninput_items[0],ninput_items[1]);

  int i = 0;
  int out_cnt = 0;

  switch(d_state){
  case HEADER:
    while(i < nr_in_chunks && out_cnt < noutput_items){
      out_new_pkt[out_cnt] = in_new_pkt[i];
      out[out_cnt++] = in[i++];
      d_byte_cnt++;
      if (HEADER_SIZE == d_byte_cnt){
        enter_payload(out);
        break;
      }
    }
    break;
  case PAYLOAD:
    while(i < nr_in_chunks && out_cnt < noutput_items){
      if (in_new_pkt[i] == 1){
        if (d_residbit_cnt > 0){
	  out[out_cnt++] = (unsigned char)(d_residbit & 0xff);
	}
        assert(d_residbit_cnt <= 8);
        enter_header();
        break;
      }
      d_residbit |= (in[i++] & ((1 << d_chunk_size)-1)) << d_residbit_cnt;
      d_residbit_cnt += d_chunk_size;
      d_byte_cnt++;

      if (d_residbit_cnt > 7){
        out[out_cnt++] = (unsigned char)(d_residbit & 0xff);
        d_residbit = d_residbit >> 8;
        d_residbit_cnt -= 8;
      }
    }
    memset(out_new_pkt,0,sizeof(char)*out_cnt);
    break;
  default:
    throw std::runtime_error("foimimo_chunk_2_byte_skip_head entered an unknown state in work");
    break;
  }

  assert(out_cnt <= noutput_items);
  assert(i <= ninput_items[0]);
  consume_each(i);
  return out_cnt;

}
