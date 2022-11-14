/* -*- c++ -*- */
/*
 * Copyright 2006 Free Software Foundation, Inc.
 *
 * Copyright (c) 2006 BBN Technologies Corp.  All rights reserved.
 * Effort sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and the Department of the Interior National Business
 * Center under agreement number NBCHC050166.
 * 
 * This file is part of GNU Radio and WiFi Localization
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


#ifndef INCLUDED_BBN_PLCP80211_BB_H
#define INCLUDED_BBN_PLCP80211_BB_H

#define LOCALIZATION_SOLVER_ENABLED 1

#include <gr_block.h>
#include <gr_msg_queue.h>
#include <bbn_scrambler_bb.h>
#if LOCALIZATION_SOLVER_ENABLED
  //#include "localization_solver.h"
  #include <sys/time.h> 
#endif

class bbn_plcp80211_bb;
typedef boost::shared_ptr<bbn_plcp80211_bb> bbn_plcp80211_bb_sptr;

bbn_plcp80211_bb_sptr bbn_make_plcp80211_bb(gr_msg_queue_sptr target_queue,
                                            bool check_crc = 0);

//bshaw added
void packet_to_solver(unsigned char* a, int b, long long c, float d);

#define SFD (0x05cf)
#define RSFD (0xf3a0)
#define MAX_PDU_LENGTH (2500)

typedef enum plcp_state_enum {
  PLCP_STATE_SEARCH_PREAMBLE,
  PLCP_STATE_SEARCH_SFD1,
  PLCP_STATE_SEARCH_SFD2,
  PLCP_STATE_SEARCH_SFD3,
  PLCP_STATE_SEARCH_RSFD1,
  PLCP_STATE_SEARCH_RSFD2,
  PLCP_STATE_SEARCH_RSFD3,
  PLCP_STATE_HDR,
  PLCP_STATE_SHDR,
  PLCP_STATE_PDU
} plcp_state_t;

//typedef 
enum plcp_current_rate_enum {
  PLCP_RATE_1MBPS,
  PLCP_RATE_2MBPS
};


//#if LOCALIZATION_SOLVER_ENABLED
#include <sys/time.h>
class NrlSolver{
  timeval starttime;
  
public:
  NrlSolver();
  ~NrlSolver();
  void passpkt(unsigned char* d_pkt_data, int d_pdu_len, 
	       long long d_packet_rx_time, float rssi);
  void setstarttime(void);
};
//#endif


/*!
 * \brief Carrier tracking PLL for QPSK
 * input: complex; output: complex
 */
class bbn_plcp80211_bb : public gr_block {
  friend bbn_plcp80211_bb_sptr bbn_make_plcp80211_bb(gr_msg_queue_sptr 
                                                     target_queue,
                                                     bool check_crc);

  bbn_plcp80211_bb (gr_msg_queue_sptr target_queue, bool check_crc);

public:
  ~bbn_plcp80211_bb ();
  void forecast (int noutput_items, gr_vector_int &ninput_items_required);
  int  general_work (int noutput_items,
                     gr_vector_int &ninput_items,
                     gr_vector_const_void_star &input_items,
                     gr_vector_void_star &output_items);


private:
  long long d_symbol_count;
  long long d_packet_rx_time;
  long long d_packet_rate;
  gr_msg_queue_sptr d_target_queue;
  bbn_scrambler_bb_sptr d_descrambler;
  plcp_state_t d_state;
  unsigned int d_data;
  int d_byte_offset;
  int d_shift;
  int d_byte_count;
  int d_pdu_len;
  plcp_current_rate_enum d_rate;
  unsigned short d_data1_in;
  unsigned int d_data2_in;
  unsigned int d_check_crc;
  unsigned char d_hdr[6];
  unsigned char bit_reverse_table[256];
  unsigned char d_pkt_data[MAX_PDU_LENGTH];
  unsigned char d_scrambler_seed;
  
#if LOCALIZATION_SOLVER_ENABLED
  NrlSolver* nrlsolver;
#endif
};

#endif
