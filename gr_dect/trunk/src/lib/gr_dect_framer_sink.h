/* -*- c++ -*- */
/*
 * Copyright 2005,2006 Free Software Foundation, Inc.
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

#ifndef INCLUDED_DECTV1_FRAMER_SINK_DECT_H
#define INCLUDED_DECTV1_FRAMER_SINK_DECT_H

#include <gr_sync_block.h>
#include <gr_msg_queue.h>

class dectv1_framer_sink_dect;
typedef boost::shared_ptr<dectv1_framer_sink_dect> dectv1_framer_sink_dect_sptr;

dectv1_framer_sink_dect_sptr 
dectv1_make_framer_sink_dect (gr_msg_queue_sptr target_queue);

/*!
 * \brief Given a stream of bits and access_code flags, assemble packets.
 * \ingroup sink
 *
 * input: stream of bytes from gr_correlate_access_code_bb
 * output: none.  Pushes assembled packet into target queue
 *
 * The framer expects a fixed length header of 2 16-bit shorts
 * containing the payload length, followed by the payload.  If the 
 * 2 16-bit shorts are not identical, this packet is ignored.  Better
 * algs are welcome.
 *
 * The input data consists of bytes that have two bits used.
 * Bit 0, the LSB, contains the data bit.
 * Bit 1 if set, indicates that the corresponding bit is the
 * the first bit of the packet.  That is, this bit is the first
 * one after the access code.
 */
class dectv1_framer_sink_dect : public gr_sync_block
{
  friend dectv1_framer_sink_dect_sptr 
  dectv1_make_framer_sink_dect (gr_msg_queue_sptr target_queue);

 private:
  enum state_t {STATE_SYNC_SEARCH, STATE_HAVE_SYNC, STATE_HAVE_HEADER};

  static const int MAX_PKT_LEN    = 4096;
  static const int HEADERBITLEN   = 32;

  gr_msg_queue_sptr  d_target_queue;		// where to send the packet when received
  state_t            d_state;
  unsigned int       d_header;			// header bits
  int		     d_headerbitlen_cnt;	// how many so far

  unsigned char      d_packet[MAX_PKT_LEN];	// assembled payload
  unsigned char	     d_packet_byte;		// byte being assembled
  int		     d_packet_byte_index;	// which bit of d_packet_byte we're working on
  int 		     d_packetlen;		// length of packet
  int                d_packet_whitener_offset;  // offset into whitener string to use
  int		     d_packetlen_cnt;		// how many so far

 protected:
  dectv1_framer_sink_dect(gr_msg_queue_sptr target_queue);

  void enter_search();
  void enter_have_sync();
  void enter_have_header(int payload_len, int whitener_offset);
  
  bool header_ok()
  {
    //todo check dect header CRC to see if the header is valid
    //return ((d_header >> 16) ^ (d_header & 0xffff)) == 0;
      return true;
  }

  void header_payload(int *len, int *offset)
  {
    //header is null in this case
    //payload is 64 bits for RFPI identities no B field, no whitening here
    //*len = (d_header >> 16) & 0x0fff;
    *offset = (d_header >> 28) & 0x000f;
    *len=480;
    //*offset=0;

  }

 public:
  ~dectv1_framer_sink_dect();

  int work(int noutput_items,
	   gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);
};

#endif /* INCLUDED_DECTV1_FRAMER_SINK_DECT_H */
