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

#include <foimimo_byte_2_chunk.h>
#include <gr_io_signature.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

#define VERBOSE 0

static const unsigned int BITS_PER_TYPE = sizeof(unsigned int) * 8;

void
foimimo_byte_2_chunk::enter_header(){
  assert(d_residbit_cnt <= 0 );
  d_byte_cnt = 0;
  d_residbit = 0;
  d_residbit_cnt = 0;
  if (VERBOSE)
    printf("byte_2_chunk@header\n");
  d_state = HEADER;
}

void
foimimo_byte_2_chunk::enter_payload(const unsigned char* header){
  d_packet_len = (unsigned int) ((header[0] & 0xff) << 8) | (header[1] & 0xff);
  d_packet_len += HEADER_SIZE;
  if (VERBOSE)
    printf("byte_2_chunk@payload with packet_len:%i\n", d_packet_len);

  d_state = PAYLOAD;
}

foimimo_byte_2_chunk_sptr
foimimo_make_byte_2_chunk(unsigned int chunk_size){
  return foimimo_byte_2_chunk_sptr (new foimimo_byte_2_chunk(chunk_size));
}

foimimo_byte_2_chunk::foimimo_byte_2_chunk(unsigned int chunk_size)
  :gr_block ("foimimo_byte_2_chunk",
      gr_make_io_signature2(2,2,sizeof(char),sizeof(char)),
      gr_make_io_signature2(2,2,sizeof(char),sizeof(char))),
  d_chunk_size(chunk_size),
  d_residbit(0),
  d_residbit_cnt(0)
{
  d_input_mask = (1 << d_chunk_size)-1;

  enter_header();
  set_output_multiple(HEADER_SIZE);
  set_relative_rate(8.0/d_chunk_size);
}

void
foimimo_byte_2_chunk::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  int input_required = (int)ceil(noutput_items*d_chunk_size/8.0);
  for(int i=0; i<ninput_items_required.size();i++){
    ninput_items_required[i] = input_required;
  }
}
int
foimimo_byte_2_chunk::general_work (int noutput_items,
                  gr_vector_int &ninput_items,
                  gr_vector_const_void_star &input_items,
                  gr_vector_void_star &output_items)
{
  //printf("noutput_items:%i, ninput_items:%i\n",noutput_items,ninput_items[0]);
  assert(noutput_items * d_chunk_size <= ninput_items[0]*8);
  const unsigned char *in = (const unsigned char*) input_items[0];
  unsigned char *out = (unsigned char*) output_items[0];
  const unsigned char *in_new_pkt = (const unsigned char *) input_items[1];
  unsigned char *out_new_pkt = (unsigned char *) output_items[1];

  int nr_of_in_items = std::min(ninput_items[0],ninput_items[1]);

  int in_byte_cnt = 0;
  int out_cnt = 0;

  switch(d_state){
  case HEADER:
    while(in_byte_cnt < nr_of_in_items && out_cnt < noutput_items){
      out_new_pkt[out_cnt] = in_new_pkt[in_byte_cnt];
      out[out_cnt++] = in[in_byte_cnt++];
      d_byte_cnt++;
      if (HEADER_SIZE == d_byte_cnt){
        enter_payload(out);
        break;
      }
    }
    break;
  case PAYLOAD:
    while(in_byte_cnt < nr_of_in_items && out_cnt < noutput_items){
      if (in_new_pkt[in_byte_cnt] == 1){
        while(d_residbit_cnt > 0 && (out_cnt < noutput_items) ){

          out[out_cnt++] = (unsigned char)((d_residbit >> (BITS_PER_TYPE - d_chunk_size)) & d_input_mask);
          d_residbit = (d_residbit << d_chunk_size);
          d_residbit_cnt -= d_chunk_size;
        }
        if(d_residbit_cnt <= 0 ){
          enter_header();

        }
        break;
      }
      if (d_residbit_cnt < d_chunk_size){ //the residual bit are too few fetch a new byte
        d_residbit |= (in[in_byte_cnt++] << (BITS_PER_TYPE-8-d_residbit_cnt)); // shift towards msb
        d_residbit_cnt += 8;
        d_byte_cnt++;
      }
      out[out_cnt++] = (unsigned char)((d_residbit >> (BITS_PER_TYPE - d_chunk_size)) & d_input_mask);
      d_residbit = (d_residbit << d_chunk_size);
      d_residbit_cnt -= d_chunk_size;

    }
    memset(out_new_pkt,0,sizeof(char)*out_cnt);
    break;
  default:
    throw std::runtime_error("foimimo_byte_2_chunk entered an unknown state in work");
    break;
  }

  assert(out_cnt <= noutput_items);
  assert(in_byte_cnt <= ninput_items[0]);
  consume_each(in_byte_cnt);
  return out_cnt;

}
