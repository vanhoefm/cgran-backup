/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2006,2007 Free Software Foundation, Inc.
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
// This is a modification of foimimo_ofdm_mapper_bcv.h from GNU Radio.

#ifndef INCLUDED_FOIMIMO_OFDM_MAPPER_SOURCE_H
#define INCLUDED_FOIMIMO_OFDM_MAPPER_SOURCE_H

#include <gr_sync_block.h>
#include <gr_message.h>
#include <gr_msg_queue.h>

class foimimo_ofdm_mapper_source;
typedef boost::shared_ptr<foimimo_ofdm_mapper_source> foimimo_ofdm_mapper_source_sptr;

foimimo_ofdm_mapper_source_sptr
foimimo_make_ofdm_mapper_source (const std::vector<gr_complex> &constellation, unsigned msgq_limit,
			 unsigned occupied_carriers, unsigned int fft_length, 
			 unsigned int output_mult = 1);

/*!
 * \brief take a stream of bytes in and map to a vector of complex
 * constellation points suitable for IFFT input to be used in an ofdm
 * modulator.  Abstract class must be subclassed with specific mapping.
 * \ingroup modulation_blk
 * \ingroup ofdm_blk
 */

class foimimo_ofdm_mapper_source : public gr_sync_block
{
  friend foimimo_ofdm_mapper_source_sptr
  foimimo_make_ofdm_mapper_source (const std::vector<gr_complex> &constellation, unsigned msgq_limit,
			   unsigned occupied_carriers, unsigned int fft_length, 
			   unsigned int output_mult);
 protected:
  foimimo_ofdm_mapper_source (const std::vector<gr_complex> &constellation, unsigned msgq_limit,
		      unsigned occupied_carriers, unsigned int fft_length,
		      unsigned int output_mult);

 private:
  std::vector<gr_complex> d_constellation;
  gr_msg_queue_sptr	d_msgq;
  gr_message_sptr	d_msg;
  unsigned		d_msg_offset;
  bool			d_eof;
  
  unsigned int 		d_occupied_carriers;
  unsigned int 		d_fft_length;
  unsigned int 		d_bit_offset;
  int			d_pending_flag;

  unsigned long  d_nbits;
  unsigned char  d_msgbytes;
  
  unsigned char d_resid;
  unsigned int d_nresid;

  std::vector<int> d_subcarrier_map;

  unsigned int d_output_mult;
  unsigned int d_output_phase;

  int randsym();

 public:
  ~foimimo_ofdm_mapper_source(void);

  gr_msg_queue_sptr	msgq() const { return d_msgq; }

  int work(int noutput_items,
	   gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);

};

#endif
