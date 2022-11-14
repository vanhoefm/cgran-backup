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

#ifndef INCLUDED_FOI_CRC32_CHECKER_SINK
#define INCLUDED_FOI_CRC32_CHECKER_SINK

#include <gr_sync_block.h>
#include <gr_message.h>
#include <gr_msg_queue.h>

class foimimo_crc32_checker_sink;
typedef boost::shared_ptr<foimimo_crc32_checker_sink> foimimo_crc32_checker_sink_sptr;

foimimo_crc32_checker_sink_sptr
foimimo_make_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue);

class foimimo_crc32_checker_sink: public gr_block
{
  friend foimimo_crc32_checker_sink_sptr
  foimimo_make_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue);

private:
  gr_msg_queue_sptr d_target_queue;
  unsigned int      d_pkt_size;

  static const int CRC32_SIZE = 4;

protected:
  foimimo_crc32_checker_sink(unsigned int pkt_size, gr_msg_queue_sptr target_queue);

public:

  void set_pkt_size(unsigned int pkt_size){d_pkt_size = pkt_size;}

  void forecast (int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items,
               gr_vector_int &ninput_items,
               gr_vector_const_void_star &input_items,
               gr_vector_void_star &output_items);
};
#endif
