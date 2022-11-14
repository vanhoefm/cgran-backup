/* -*- c++ -*- */
/*
 * Copyright 2008 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
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
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_GR_HDLC_ROUTER_SINK_B_H
#define INCLUDED_GR_HDLC_ROUTER_SINK_B_H

#include <gr_sync_block.h>
#include <omnithread.h>
#include <ppio.h>

class gr_hdlc_router_sink_b;
typedef boost::shared_ptr<gr_hdlc_router_sink_b> gr_hdlc_router_sink_b_sptr;

gr_hdlc_router_sink_b_sptr gr_make_hdlc_router_sink_b(int dlci);

/*!
 * \brief Extract and send IP packets that are carried in a synchronous hdlc 
 * bitstream that uses Multi-Protocol over Frame Relay (MPoFR) to encapsulate 
 * the IP packets.
 * \ingroup sink
 */

class gr_hdlc_router_sink_b : public gr_sync_block
{
  friend gr_hdlc_router_sink_b_sptr gr_make_hdlc_router_sink_b(int dlci);

 private:
  // CONSTANTS ------------------
  static const int           SUCCESS      = 1;
  static const int           FAIL         = 0;
  static const int           BIT_BUF_MAX  = 78644; // Allow for up to 20% bitstuffing
  static const int           FRAME_BUF_MAX= 9831;  // Allow for no stuffed bits
  static const int           FRAME_MAX    = 8192;
  static const unsigned char FLAG         = 0x7E;
  static const int           HUNT         = 0;
  static const int           IDLE         = 1;
  static const int           FRAMING      = 2;

  // Privete Attributes ---------

  // Frame Relay's Data Link Channel Indicator
  int            d_dlci; 

  // State machine state info
  int            d_state;
  unsigned char  d_byte;  // Accumulator for building a flag byte from bits
  int            d_accumulated_bits;        // Bit counter for d_byte
  unsigned char  d_bit_buf[BIT_BUF_MAX];
  int            d_bit_buf_size;
  int            d_consecutive_one_bits;

  // Data statistics
  int            d_flag_cnt;
  int            d_good_frame_cnt;
  int            d_good_byte_cnt;
  int            d_good_dlci_cnt;

  // Error statistics
  int            d_crc_err_cnt;
  int            d_abort_cnt;
  int            d_seven_ones_cnt;
  int            d_non_align_cnt;
  int            d_giant_cnt;
  int            d_runt_cnt;

  // Private Methods ----------

  unsigned short crc16(unsigned char *data_p, 
                       unsigned short length);
 
  int crc_valid(int frame_size, unsigned char * frame);


  void route_packet(int             hdlc_frame_size, 
                    unsigned char * hdlc_frame);

  int unstuff(int             bit_buf_size, 
              unsigned char * bit_buf, 
              int *           frame_buf_size, 
              unsigned char * frame_buf);

  void hdlc_state_machine(unsigned char next_bit);


 protected:
  gr_hdlc_router_sink_b(int dlci);

 public:
  ~gr_hdlc_router_sink_b();


  int work(int noutput_items,
	   gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);

  int dlci()        const { return d_dlci; }
  int flags()       const { return d_flag_cnt; }
  int frames()      const { return d_good_frame_cnt; }
  int bytes()       const { return d_good_byte_cnt; }
  int dlci_frames() const { return d_good_dlci_cnt; }
  int crc_errs()    const { return d_crc_err_cnt; }
  int aborts()      const { return d_abort_cnt; }
  int seven_ones()  const { return d_seven_ones_cnt; }
  int non_aligned() const { return d_non_align_cnt; }
  int giants()      const { return d_giant_cnt; }
  int runts()       const { return d_runt_cnt; }
  void clear_counters()   {
                            d_flag_cnt = 0;
                            d_good_frame_cnt = 0;
                            d_good_byte_cnt = 0;
                            d_good_dlci_cnt = 0;
                            d_crc_err_cnt = 0;
                            d_abort_cnt = 0;
                            d_seven_ones_cnt = 0;
                            d_non_align_cnt = 0;
                            d_giant_cnt = 0;
                            d_runt_cnt = 0;
                          }

};


#endif /* INCLUDED_GR_HDLC_ROUTER_SINK_B_H */
