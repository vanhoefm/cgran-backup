/* -*- c++ -*- */
/*
 * Copyright 2006,2009 Free Software Foundation, Inc.
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

#include <stt_tag_reader_msg.h>
#include <gr_io_signature.h>
#include <string.h>
#include <stdexcept>
#include "errno.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <cstdio>
#include <gr_tags.h>




stt_tag_reader_msg_sptr
stt_make_tag_reader_msg(size_t itemsize, gr_msg_queue_sptr msgq)
{
  return gnuradio::get_initial_sptr(new stt_tag_reader_msg(itemsize, msgq));
}

stt_tag_reader_msg::stt_tag_reader_msg(size_t itemsize, gr_msg_queue_sptr msgq)
  : gr_block ("tag_msg_source",
	      gr_make_io_signature (1, 1, itemsize),
	      gr_make_io_signature (1, 1, itemsize)),
    d_itemsize(itemsize),
    d_msgq(msgq)
{
}

bool
stt_tag_reader_msg::check_topology(int ninputs, int noutputs)
{
  return ninputs == noutputs;
}

int
stt_tag_reader_msg::general_work(int noutput_items,
		      gr_vector_int &ninput_items,
		      gr_vector_const_void_star &input_items,
		      gr_vector_void_star &output_items)
{
  const uint8_t *in = (const uint8_t *) input_items[0];
  uint8_t *out = (uint8_t *) output_items[0];
  int n = std::min<int>(ninput_items[0], noutput_items);
  int j = 0;

  uint64_t start_N = nitems_read(0);
  uint64_t end_N = start_N + (uint64_t)(n);
  pmt::pmt_t bkey = pmt::pmt_string_to_symbol("time");	// TODO: make a parameter
  std::vector<gr_tag_t> all_tags;

  get_tags_in_range(all_tags, 0, start_N, end_N);
  
  std::sort(all_tags.begin(), all_tags.end(), gr_tag_t::offset_compare);

  // create a message for each tag
  std::vector<gr_tag_t>::iterator vitr = all_tags.begin();
  for(;(vitr != all_tags.end());vitr++)
  {
    //if (pmt::pmt_eqv((*vitr).key, bkey))
    {
       pmt::pmt_t tag_value = (*vitr).value;
       if ( pmt::pmt_is_symbol(tag_value) )
       {
          std::string tag_str = pmt::pmt_symbol_to_string(tag_value);
          gr_message_sptr msg = gr_make_message_from_string(tag_str.c_str());
	  d_msgq->handle(msg);
       }
       else 
       { 
       	  char cstr[50];
          if ( pmt::pmt_is_number(tag_value) )
          {
	     uint64_t tag_val = pmt::pmt_to_uint64(tag_value);
             sprintf ( cstr, "%lld\n", tag_val );
          }
          else
          {
             sprintf ( cstr, "unknown tag value" );
          }
          gr_message_sptr msg = gr_make_message_from_string(cstr);
          d_msgq->handle(msg);
       }
    }
  } 
 
  // copy data
  memcpy(out, in, n*d_itemsize);
  j = n;

  consume_each(n);
  return j;
}
