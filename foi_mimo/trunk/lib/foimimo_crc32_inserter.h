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

#ifndef INCLUDED_FOI_CRC32_INSERTER
#define INCLUDED_FOI_CRC32_INSERTER

#include <gr_sync_block.h>
#include <gr_message.h>
#include <gr_msg_queue.h>

class foimimo_crc32_inserter;
typedef boost::shared_ptr<foimimo_crc32_inserter> foimimo_crc32_inserter_sptr;

foimimo_crc32_inserter_sptr
foimimo_make_crc32_inserter(unsigned int bytes_per_pkt, unsigned int msgq_limit);

class foimimo_crc32_inserter: public gr_sync_block
{
  friend foimimo_crc32_inserter_sptr
  foimimo_make_crc32_inserter(unsigned int bytes_per_pkt, unsigned int msgq_limit);

protected:
  foimimo_crc32_inserter(unsigned int bytes_per_pkt, unsigned int msgq_limit);


private:
  gr_msg_queue_sptr    d_msgq;
  gr_message_sptr      d_msg;
  bool                 d_eof;

  unsigned int d_bytes_per_packet;

  static const int HEADER_SIZE = 4; // Header size in bytes. 32 bit the header is 16 bit but copied.
                                    // 12 bit for pkt length
  static const int CRC32_SIZE = 4;  // the crc32 size is 4 byte

  unsigned int          d_nresidbyte;
  unsigned char*        d_residbyte;
  int                   d_msg_offset;

  unsigned int make_header(unsigned int bytes_per_packet);

public:
  ~foimimo_crc32_inserter(void);
  gr_msg_queue_sptr   msgq() const { return d_msgq; }

  int work(int noutput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

};


#endif /* INCLUDED_foimimo_CRC32_INSERTER */
