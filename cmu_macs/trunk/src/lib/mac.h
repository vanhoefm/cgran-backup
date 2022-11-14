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

#ifndef INCLUDED_MAC_H
#define INCLUDED_MAC_H

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

#include <gmsk_symbols.h>

class mac;

class mac : public mb_mblock
{

 protected:
  // The state is used to determine how to handle incoming messages and of
  // course, the state of the MAC protocol.
  enum usrp_state_t {
    OPENING_USRP,
    ALLOCATING_CHANNELS,
    CONNECTED,
    DEALLOCATING_CHANNELS,
    CLOSING_USRP,
  };
  usrp_state_t	d_usrp_state;

  enum channel_type {
    RX_CHANNEL,
    TX_CHANNEL,
  };

  // Port to the PHY, MAC must make the connection
  mb_port_sptr      d_phy_cs;

  // Ports to connect to usrp_server (us)
  mb_port_sptr      d_us_tx, d_us_rx, d_us_cs;
  
  // The channel numbers assigned for use
  pmt_t d_us_rx_chan, d_us_tx_chan;

  // USRP parameters
  long d_usrp_decim;
  long d_usrp_interp;

  pmt_t d_which_usrp;
  
  bool d_rx_enabled;

  // FPGA regs
  enum FPGA_REGISTERS {
    REG_CS_THRESH = 51,
    REG_CS_DEADLINE = 52
  };
  
  virtual void handle_mac_message(mb_message_sptr msg);   // MAC overridable
  virtual void usrp_initialized();                        // MAC overridable
  virtual void packet_transmitted(pmt_t data);            // MAC overridable
  void enable_rx();
  void disable_rx();
  
 public:
  mac(mb_runtime *rt, const std::string &instance_name, pmt_t user_arg);
  ~mac();

  // Top level message dispatcher to separate the USRP init and teardown so that
  // every MAC does not have to keep re-writing this code.
  void handle_message(mb_message_sptr msg);

 private:
  void define_usrp_ports();
  void initialize_usrp();
  void handle_usrp_message(mb_message_sptr msg);

  void open_usrp();
  void open_usrp_response(pmt_t data, bool success);
  void close_usrp();
  
  void deallocate_channels();
  void deallocate_channels_response(pmt_t data, channel_type chan, bool success);
  
  void allocate_channels();
  void allocate_channels_response(pmt_t data, channel_type chan, bool success);

  void transmit_pkt(pmt_t data);
};

#endif // INCLUDED_MAC_H

