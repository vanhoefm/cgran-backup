/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2004,2010 Free Software Foundation, Inc.
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
// This is a modification of trellis_encoder_bb.cc from GNU Radio.


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_trellis_encoder_bb.h>
#include <gr_io_signature.h>
#include <gr_log2_const.h>
#include <stdio.h>
#include <math.h>

#define VERBOSE 0

void
foimimo_trellis_encoder_bb::enter_header(){
  d_byte_cnt = 0;
  d_ST = 0;
  if (VERBOSE)
    printf("encoder@header\n");
  d_state = HEADER;
}

void
foimimo_trellis_encoder_bb::enter_payload(const unsigned char* header){
  d_packet_len = (unsigned int) ((header[0] & 0xff) << 8) | (header[1] & 0xff);

  d_packet_len = ceil(d_packet_len*8.0/log2(d_FSM.I())) + HEADER_SIZE;
  if (VERBOSE)
    printf("encoder@payload with packet_len:%i\n", d_packet_len);
  d_state = PAYLOAD;
}

foimimo_trellis_encoder_bb_sptr
foimimo_make_trellis_encoder_bb (const fsm &FSM, int ST)
{
  return foimimo_trellis_encoder_bb_sptr (new foimimo_trellis_encoder_bb (FSM,ST));
}

foimimo_trellis_encoder_bb::foimimo_trellis_encoder_bb (const fsm &FSM, int ST)
  : gr_sync_block ("encoder_bb",
                    gr_make_io_signature2(2,2,sizeof(char),sizeof(char)),
                    gr_make_io_signature2(2,2,sizeof(char),sizeof(char))),
    d_FSM (FSM),
    d_ST (ST)
{

  // generate input bit mask
  d_input_mask = 0;
  unsigned int tmp = 0;
  while (d_FSM.I() > pow(2,tmp)){
    d_input_mask |= (1<< tmp++);
  }
  enter_header();
}

int 
foimimo_trellis_encoder_bb::work (int noutput_items,
                              gr_vector_const_void_star &input_items,
                              gr_vector_void_star &output_items)
{
  const unsigned char *in = (const unsigned char *) input_items[0];
  unsigned char *out = (unsigned char *) output_items[0];
  const unsigned char *in_new_pkt = (const unsigned char *) input_items[1];
  unsigned char *out_new_pkt = (unsigned char *) output_items[1];

  int out_byte_cnt = 0;
  int in_byte_cnt = 0;

  switch(d_state){
  case HEADER:
    while(out_byte_cnt < noutput_items){
      out_new_pkt[out_byte_cnt] = in_new_pkt[in_byte_cnt];
      out[out_byte_cnt++] = in[in_byte_cnt++];
      d_byte_cnt++;
      if (HEADER_SIZE == d_byte_cnt){
        enter_payload(out);
        break;
      }
    }
    break;
  case PAYLOAD:
    while(out_byte_cnt < noutput_items){
      if (in_new_pkt[in_byte_cnt] == 1){
          enter_header();
          break;
      }
      out_new_pkt[out_byte_cnt] = 0;
      // encode
      out[out_byte_cnt++] =(unsigned char) d_FSM.OS()[d_ST*d_FSM.I()+in[in_byte_cnt]]; // direction of time?
      d_ST = (int) d_FSM.NS()[d_ST*d_FSM.I()+in[in_byte_cnt++]];
      d_byte_cnt++;

    }
    break;
  default:
    throw std::runtime_error("foimimo_trellis_encoder entered an unknown state in work");
    break;
  }

  return out_byte_cnt;
}

