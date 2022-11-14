/* -*- c++ -*- */
/*
 * Copyright 2004,2006 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <gr_dect_correlate_access_code.h>
#include <gr_io_signature.h>
#include <stdexcept>
#include <gr_count_bits.h>
#include <sys/time.h>

#define VERBOSE 0


dectv1_correlate_access_code_dect_sptr
dectv1_make_correlate_access_code_dect (const std::string &access_code, int threshold)
{
  return dectv1_correlate_access_code_dect_sptr (new dectv1_correlate_access_code_dect (access_code, threshold));
}


dectv1_correlate_access_code_dect::dectv1_correlate_access_code_dect (
  const std::string &access_code, int threshold)
  : gr_sync_block ("correlate_access_code_dect",
		   gr_make_io_signature (1, 1, sizeof(char)),
		   gr_make_io_signature (1, 1, sizeof(char))),
    d_data_reg(0), d_flag_reg(0), d_flag_bit(0), d_mask(0),
    d_threshold(threshold)

{
  if (!set_access_code(access_code)){
    fprintf(stderr, "dectv1_correlate_access_code_dect: access_code is > 64 bits\n");
    throw std::out_of_range ("access_code is > 64 bits");
  }
}

dectv1_correlate_access_code_dect::~dectv1_correlate_access_code_dect ()
{
}

bool
dectv1_correlate_access_code_dect::set_access_code(
  const std::string &access_code)
{
  unsigned len = access_code.length();	// # of bytes in string
  bit_count=0;
  bit_count_prev=0;
  if (len > 64)
    return false;

  // set len top bits to 1.
  d_mask = ((~0ULL) >> (64 - len)) << (64 - len);

  d_flag_bit = 1LL << (64 - len);	// Where we or-in new flag values.
                                        // new data always goes in 0x0000000000000001
  d_access_code = 0;
  for (unsigned i=0; i < 64; i++){
    d_access_code <<= 1;
    if (i < len)
      d_access_code |= access_code[i] & 1;	// look at LSB only
  }

  return true;
}

int
dectv1_correlate_access_code_dect::work (int noutput_items,
				   gr_vector_const_void_star &input_items,
				   gr_vector_void_star &output_items)
{
  const unsigned char *in = (const unsigned char *) input_items[0];
  unsigned char *out = (unsigned char *) output_items[0];
  static struct timeval zeit;

  for (int i = 0; i < noutput_items; i++){

    // compute output value
    unsigned int t = 0;
    //fprintf(stdout, "output1 %ho\n", t);
    t |= ((d_data_reg >> 63) & 0x1) << 0;
    //fprintf(stdout, "output2 %ho\n", t);
    //fprintf(stdout, "d flag reg: %llx\n", d_flag_reg);
    t |= ((d_flag_reg >> 63) & 0x1) << 1;	// flag bit
//    t |= ((d_time_reg1 >> 63) & 0x1) << 2;	// time bit 1
//   t |= ((d_time_reg2 >> 63) & 0x1) << 3;	// time bit 2
    //fprintf(stdout, "output3 %ho\n", t);
    out[i] = t;
    bit_count++;
    //fprintf(stdout, "output %hho\n", out[i]);

    
    // compute hamming distance between desired access code and current data
    unsigned long long wrong_bits = 0;
    unsigned int nwrong = d_threshold+1;
    int new_flag = 0;

    wrong_bits  = (d_data_reg ^ d_access_code) & d_mask;
    nwrong = gr_count_bits64(wrong_bits);

    // test for access code with up to threshold errors
    new_flag = (nwrong <= d_threshold);

	//if(new_flag) {  
	//	fprintf(stdout, "target code : %llx\n", d_access_code);
	//	fprintf(stdout, "dect code  found: %llx\n", d_data_reg);
	//	}

#if VERBOSE
    if(new_flag) {
      fprintf(stderr, "access code found: %llx\n", d_access_code);
    }
    else {
      fprintf(stderr, "%llx  ==>  %llx\n", d_access_code, d_data_reg);
    }
#endif

    // shift in new data and new flag
    d_data_reg = (d_data_reg << 1) | (in[i] & 0x1);
    d_flag_reg = (d_flag_reg << 1);
 //   d_time_reg1 = (d_time_reg1 << 1);
 //   d_time_reg2 = (d_time_reg2 << 1);
    if (new_flag) {
      //gettimeofday(&zeit,NULL);
      //fprintf(stderr, "no output items%u\n", noutput_items);
      //unsigned int t1 = 0;
      //fprintf(stdout, "time stamp 1 %ld \n", tprev);
      //fprintf(stdout, "time stamp 2 %ld \n", zeit.tv_usec/416);
      //fprintf(stdout, "frame length %ld \n", zeit.tv_usec/417-tprev);
      //tprev=zeit.tv_usec/417;
      //fprintf(stdout, "super frame number %lld \n", (bit_count/11520)%16);
      //fprintf(stdout, "frame number %lld \n", (bit_count/480)%24);
      if (bit_count >= 184320)
	{bit_count=0;}

      //bit_count_prev=bit_count/480;
      //fprintf(stdout, "d flag regA: %llx\n", d_flag_reg);
      d_flag_reg |= d_flag_bit;
//      if (zeit.tv_usec/416<1000){
//	      d_time_reg1|= d_flag_bit;}
  //    if (zeit.tv_usec/416>=1000 && zeit.tv_usec/416<2000){
	//      d_time_reg2|= d_flag_bit;}
      //fprintf(stderr, "flag %llx\n", d_flag_reg);
      //t1 |= ((d_data_reg >> 63) & 0x1) << 0;
      //t1 |= ((d_flag_reg >> 63) & 0x1) << 1;	// flag bit
      //fprintf(stdout, "time reg 1: %llx\n", d_time_reg1);
      //fprintf(stdout, "time reg 2: %llx\n", d_time_reg2);
      //fprintf(stdout, "d flag regB: %llx\n", d_flag_reg);

      //fprintf(stderr, "data byte %lx\n", t1);
    }
  }
	
  return noutput_items;
}
  
