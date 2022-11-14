/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
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
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef INCLUDED_CMAC_H
#define INCLUDED_CMAC_H

#include <mblock/mblock.h>
#include <mblock/runtime.h>
#include <mblock/protocol_class.h>
#include <mblock/exception.h>
#include <mblock/msg_queue.h>
#include <mblock/message.h>
#include <mblock/msg_accepter.h>
#include <mblock/class_registry.h>
#include <pmt.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include <symbols_usrp_server_cs.h>
#include <symbols_usrp_channel.h>
#include <symbols_usrp_low_level_cs.h>
#include <symbols_usrp_tx.h>
#include <symbols_usrp_rx.h>

#include <cmac_symbols.h>
#include <gmsk_symbols.h>
#include <cmac_framer.h>
#include <mac_symbols.h>

#include <mac.h>

class cmac;

class cmac : public mac
{

  enum cmac_state_t {
    INIT_CMAC,
    ACK_WAIT,
    SEND_ACK,
    IDLE,
  };
  cmac_state_t	d_state;
  
  enum state_t {
    SYNC_SEARCH,
    WAIT_HEADER,
    HAVE_HEADER,
    WAIT_PAYLOAD,
    HAVE_PAYLOAD
  };
  state_t	d_framer_state;

  // Mainly for debugging
  bool d_verbose_frames;
  unsigned long d_nframes_recvd;

  // Ports used for applications to connect to this block
  mb_port_sptr		  d_tx, d_rx, d_cs;

  pmt_t d_ack_timeout;
  pmt_t d_last_frame;
  long d_last_frame_dst;

  bool d_carrier_sense;
  long d_cs_thresh;
  long d_cs_deadline;

  // Local user address
  long d_local_address;

  // Framer
  d_frame_hdr_t d_frame_hdr;
  d_frame_hdr_t d_cframe_hdr;
  std::vector<unsigned char> d_hdr_bits;
  std::vector<unsigned char> d_payload_bits;
  unsigned long d_frame_timestamp;
  unsigned long d_seq_num;
  
 protected:
  void handle_mac_message(mb_message_sptr msg);   // overriding MAC method
  void usrp_initialized();                        // overriding MAC method

 public:
  cmac(mb_runtime *rt, const std::string &instance_name, pmt_t user_arg);
  ~cmac();
  static int max_frame_size() {
    return(MAX_FRAME_SIZE);
  }
  static int max_frame_payload() {
    return(MAX_FRAME_SIZE-sizeof(d_frame_hdr_t));
  }

 private:
  // CMAC initialization
  void define_mac_ports();
  void initialize_cmac();

  // Crucial CSMA methods
  void packet_transmitted(pmt_t data);
  void incoming_data(pmt_t data);
  void incoming_frame(pmt_t data);
  void build_and_send_ack(long dst);
  void handle_ack(long src, long dst);
  
  // Carrier sense functionality
  void set_carrier_sense(bool toggle, long threshold, long deadline, pmt_t invocation);

  // Framer
  void framer(const std::vector<unsigned char> input, pmt_t demod_properties);
  void framer_calculate_timestamp(unsigned long timestamp, int bit, int nbits, long sps, long bps);
  void framer_found_sync();
  void framer_new_header_bit(unsigned char bit);
  void framer_new_payload_bit(unsigned char bit);
  void framer_have_header();
  void framer_have_payload();
  void framer_have_frame(pmt_t uvec);
  void build_frame(pmt_t data);
};

#endif // INCLUDED_CMAC_H
