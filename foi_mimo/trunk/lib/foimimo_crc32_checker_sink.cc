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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_crc32_checker_sink.h>
#include <gr_io_signature.h>
#include <gr_crc32.h>
#include <string.h>
#include <stdio.h>

foimimo_crc32_checker_sink_sptr
foimimo_make_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue)
{
  return foimimo_crc32_checker_sink_sptr(new foimimo_crc32_checker_sink(pkt_size, target_queue));
}

foimimo_crc32_checker_sink::foimimo_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue)
  :gr_block("foimimo_crc32_checker_sink",
                  gr_make_io_signature2(2,2,sizeof(char),sizeof(char)),
                  gr_make_io_signature(0,0,0)),
                  d_pkt_size(pkt_size),
                  d_target_queue(target_queue)
{

}

void
foimimo_crc32_checker_sink::forecast (int noutput_items,
                                  gr_vector_int &ninput_items_required)
{
  unsigned int items_req = d_pkt_size;
  ninput_items_required[0] = items_req;
  ninput_items_required[1] = items_req;
}

int
foimimo_crc32_checker_sink::general_work(int noutput_items,
    gr_vector_int &ninput_items,
    gr_vector_const_void_star &input_items,
    gr_vector_void_star &output_items)
{
  unsigned char *in = (unsigned char*) input_items[0];
  unsigned char *in_new_pkt = (unsigned char*) input_items[1];
  unsigned int crc32 = 0;

  unsigned int recv_crc32 = 0;
  int packet_end = 0;
  while(in_new_pkt[packet_end]==0){
    packet_end++;
  }
  unsigned int payload_size = packet_end+1 -CRC32_SIZE;

  crc32 = gr_crc32(&in[0], payload_size);
  recv_crc32 = (in[payload_size] << 24) | (in[payload_size+1] << 16) | (in[payload_size+2] << 8) | in[payload_size+3];
  
  gr_message_sptr msg =
      gr_make_message(0,crc32 == recv_crc32,0,payload_size);
  memcpy(msg->msg(), (void*)in, payload_size);
  d_target_queue->insert_tail(msg);
  msg.reset();

  consume_each(payload_size + CRC32_SIZE);
  return 0;
}
