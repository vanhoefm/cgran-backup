/* -*- c++ -*- */
/*
 * Copyright 2006 Free Software Foundation, Inc.
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

#ifndef INCLUDED_LISTENER_FIND_CW
#define INCLUDED_LISTENER_FIND_CW

#include <gr_message.h>
#include <gr_msg_queue.h>
#include <gr_sync_block.h>
#include <gr_feval.h>
#include <listener_reader_monitor_cmd_gate.h>

class listener_find_CW;
typedef boost::shared_ptr<listener_find_CW> listener_find_CW_sptr;


listener_find_CW_sptr
listener_make_find_CW(unsigned int vlen,
			float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor);
			

class listener_find_CW : public gr_sync_block
{
  friend listener_find_CW_sptr
  listener_make_find_CW(unsigned int vlen,
			float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor);

  int d_vlen;
  float d_max[1024];
  float d_samp_rate, d_usrp_freq;
  listener_reader_monitor_cmd_gate_sptr d_reader_monitor;
  int max_bin_index_new, max_bin_index_old;
  float max_abs;
  float reader_freq;
  enum state_t { BIN_STAT , OK };
  
  listener_find_CW(unsigned int vlen,
			float usrp_freq, float samp_rate, listener_reader_monitor_cmd_gate_sptr reader_monitor);
  
  int max_min(float * buffer, int len);
  void send_stat(int max_bin_index);

 
public:
  ~listener_find_CW();
  
  int work(int noutput_items, 
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);
  
};

#endif
