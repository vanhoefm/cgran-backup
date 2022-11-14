/* -*- c++ -*- */
/*
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

GR_SWIG_BLOCK_MAGIC(gr,ofdm_mapper_bcv);

gr_ofdm_mapper_bcv_sptr 
gr_make_ofdm_mapper_bcv (const std::vector<gr_complex> &constellation,
			 unsigned int msgq_limit,
			 unsigned int bits_per_symbol, 
			 unsigned int fft_length,
             std::string carrier_map);   // linklab, add carrier_map to indicate subcarrier allocation


class gr_ofdm_mapper_bcv : public gr_sync_block
{
 protected:
  gr_ofdm_mapper_bcv (const std::vector<gr_complex> &constellation,
		      unsigned int msgq_limit,
		      unsigned int bits_per_symbol,
		      unsigned int fft_length,
              std::string carrier_map);  // linklab, add carrier_map to indicate subcarrier allocation
  
 public:
  gr_msg_queue_sptr msgq();
  
  int work(int noutput_items,
	   gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);

  // linklab, initialize subcarrier allocation 
  void initialize();
  
  // linklab, reset subcarrier allocation 
  void reset_carrier_map(std::string new_carrier_map);
};
