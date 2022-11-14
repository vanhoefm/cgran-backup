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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <ftw_repetition.h>
#include <gr_io_signature.h>
#include <stdexcept>
#include <iostream>
#include <string.h>



ftw_repetition_sptr
ftw_make_repetition(int symbol_length, int repetition, int N_symbols)
{
  return ftw_repetition_sptr(new ftw_repetition(symbol_length, repetition, N_symbols));
}

ftw_repetition::ftw_repetition (int symbol_length, int repetition, int N_symbols) : 
			gr_block("repetition",gr_make_io_signature(1, 1, sizeof(gr_complex)*(symbol_length)),
			gr_make_io_signature(1, 1, sizeof(gr_complex)*symbol_length)),d_symbol_length(symbol_length),
			d_repetition(repetition),d_N_symbols(N_symbols)
{
  set_output_multiple((d_N_symbols + 5 + 13));
}


ftw_repetition::~ftw_repetition(){
} 

int counter=0;

int ftw_repetition::general_work (int noutput_items,
				  gr_vector_int &ninput_items_v,
				  gr_vector_const_void_star &input_items,
				  gr_vector_void_star &output_items)
{
  const gr_complex *in_sym = (const gr_complex *) input_items[0];

  gr_complex *out_sym = (gr_complex *) output_items[0];
  int no = 0;	// number of output items
  int ni = 0;	// number of items read from input

  while(no < noutput_items){
    memcpy(&out_sym[no * d_symbol_length],
    &in_sym[(ni %(d_N_symbols + 5 + 13)) * d_symbol_length],
    d_symbol_length * sizeof(gr_complex));
    no++;			
    ni++;
  }
  counter++;
  if ((counter==d_repetition)&&(d_repetition!=0))
    consume_each(d_N_symbols + 5 + 13);
  return no ;
}

