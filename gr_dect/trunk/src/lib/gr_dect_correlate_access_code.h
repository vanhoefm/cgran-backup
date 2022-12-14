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

#ifndef INCLUDED_DECTV1_CORRELATE_ACCESS_CODE_DECT_H
#define INCLUDED_DECTV1_CORRELATE_ACCESS_CODE_DECT_H

#include <gr_sync_block.h>
#include <string>

class dectv1_correlate_access_code_dect;
typedef boost::shared_ptr<dectv1_correlate_access_code_dect> dectv1_correlate_access_code_dect_sptr;

/*!
 * \param access_code is represented with 1 byte per bit, e.g., "010101010111000100"
 * \param threshold maximum number of bits that may be wrong
 */
dectv1_correlate_access_code_dect_sptr 
dectv1_make_correlate_access_code_dect (const std::string &access_code, int threshold);

/*!
 * \brief Examine input for specified access code, one bit at a time.
 * \ingroup block
 *
 * input:  stream of bits, 1 bit per input byte (data in LSB)
 * output: stream of bits, 2 bits per output byte (data in LSB, flag in next higher bit)
 *
 * Each output byte contains two valid bits, the data bit, and the
 * flag bit.  The LSB (bit 0) is the data bit, and is the original
 * input data, delayed 64 bits.  Bit 1 is the
 * flag bit and is 1 if the corresponding data bit is the first data
 * bit following the access code. Otherwise the flag bit is 0.
 */
class dectv1_correlate_access_code_dect : public gr_sync_block
{
  friend dectv1_correlate_access_code_dect_sptr 
  dectv1_make_correlate_access_code_dect (const std::string &access_code, int threshold);
 private:
  unsigned long long d_access_code;	// access code to locate start of packet
                                        //   access code is left justified in the word
  unsigned long long d_data_reg;	// used to look for access_code
  unsigned long long d_flag_reg;	// keep track of decisions
  unsigned long long d_flag_bit;	// mask containing 1 bit which is location of new flag
  unsigned long long d_time_reg1;	//mask containing the most significant 6 bits of timestamp
  unsigned long long d_time_reg2;	//mask containing the least significant 6 bits of timestamp
  unsigned long long d_mask;		// masks access_code bits (top N bits are set where
                                        //   N is the number of bits in the access code)
  unsigned int	     d_threshold;	// how many bits may be wrong in sync vector
  long tprev;
  unsigned long long bit_count;
  unsigned long long bit_count_prev;

 protected:
  dectv1_correlate_access_code_dect(const std::string &access_code, int threshold);

 public:
  ~dectv1_correlate_access_code_dect();

  int work(int noutput_items,
	   gr_vector_const_void_star &input_items,
	   gr_vector_void_star &output_items);

  
  /*!
   * \param access_code is represented with 1 byte per bit, e.g., "010101010111000100"
   */
  bool set_access_code (const std::string &access_code);
};

#endif /* INCLUDED_DECTV1_CORRELATE_ACCESS_CODE_DECT_H */
