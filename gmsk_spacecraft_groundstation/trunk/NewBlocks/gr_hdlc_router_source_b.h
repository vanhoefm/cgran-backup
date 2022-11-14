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

#ifndef INCLUDED_GR_HDLC_ROUTER_SOURCE_B_H
#define INCLUDED_GR_HDLC_ROUTER_SOURCE_B_H

#include <gr_sync_block.h>
#include <omnithread.h>


class gr_hdlc_router_source_b;
typedef boost::shared_ptr<gr_hdlc_router_source_b> gr_hdlc_router_source_b_sptr;

gr_hdlc_router_source_b_sptr gr_make_hdlc_router_source_b(int   dlci,
                                                          char* local_addr,
                                                          char* remote_addr);

/*!
 * \brief Receive IP packets fron a TUN pseudo network device, and compose
 * and output a bitstream that uses Multi-Protocol over Frame Relay (MPoFR) 
 * to encapsulate the IP packets.
 * \ingroup source
 */

class gr_hdlc_router_source_b : public gr_sync_block
{
  friend gr_hdlc_router_source_b_sptr gr_make_hdlc_router_source_b(int   dlci, 
                                                                   char* local_addr,
                                                                   char* remote_addr);

 private:

  // Private nested Class

  class fifo_c
  {
    private:
      static const int  FIFO_SIZE = 76651;     // (8192 x 8 x 1.2) + 7
      unsigned char     d_ring_buf[FIFO_SIZE];
      int               d_pop_index;           // next position to pop from
      int               d_push_index;          // next position to push to
    public:
      fifo_c():d_pop_index(0),d_push_index(0) {};
      ~fifo_c() {};
      void push(unsigned char c) 
        {
          d_ring_buf[d_push_index] = c;
          d_push_index = (d_push_index + 1) % FIFO_SIZE;
        };
      unsigned char pop()
        {
          unsigned char c;
          c = d_ring_buf[d_pop_index];
          d_pop_index = (d_pop_index + 1) % FIFO_SIZE;
          return c;
        };
      int empty() { return d_pop_index==d_push_index; };
      int full()  { return (d_push_index+1)%FIFO_SIZE == d_pop_index; };
      int space_left() { return FIFO_SIZE - 1 - (d_push_index - d_pop_index)%FIFO_SIZE; };
  };


  // Private CONSTANTS ------------------
  static const int           LINUX        = 0;
  static const int           OSX          = 1;
  static const int           FRAME_MAX    = 8192;
  static const int           STR_MAX      = 256;   // Limit string lengths
  static const unsigned char FLAG         = 0x7E;

  // Privete Attributes ---------

  // Frame Relay's Data Link Channel Indicator
  int            d_dlci; 

  // MPoFR Header
  unsigned char  d_header[4];

  // IP addresses for point-to-point TUN interface
  char           d_local_addr[STR_MAX];
  char           d_remote_addr[STR_MAX];

  // Bitstuffing state variable
  int            d_consecutive_one_bits;

  // File descriptor for tun device
  int            d_tun_fd;

  // Bitstream fifo
  fifo_c         d_fifo;  

  // Data statistics
  int            d_flag_cnt;
  int            d_frame_cnt;
  int            d_byte_cnt;


  // Private Methods ----------

  unsigned short crc16(unsigned char *data_p, 
                       unsigned short length);
  void push_flag(void);
  void bitstuff_byte(unsigned char byte);
  void bitstuff_and_frame_packet(unsigned char * frame_buf,
                                 int             frame_size);
  int read_packet(unsigned char * packet_buf);
  void encapsulate_incoming_packet(void);

 protected:
  gr_hdlc_router_source_b(int dlci, char* local_addr, char* remote_addr);

 public:
  ~gr_hdlc_router_source_b();


  int work(int noutput_items,
	   gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);

  int dlci()   const    { return d_dlci; }
  int flags()  const    { return d_flag_cnt; }
  int frames() const    { return d_frame_cnt; }
  int bytes()  const    { return d_byte_cnt; }
  void clear_counters() { d_flag_cnt = 0; d_frame_cnt = 0; d_byte_cnt = 0; }
};


#endif /* INCLUDED_GR_HDLC_ROUTER_SOURCE_B_H */
