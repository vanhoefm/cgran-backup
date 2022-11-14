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

#ifndef INCLUDED_GR_COPY_H
#define INCLUDED_GR_COPY_H

#include <gr_block.h>
#include <gr_message.h>
#include <gr_msg_queue.h>

class stt_tag_reader_msg;
typedef boost::shared_ptr<stt_tag_reader_msg> stt_tag_reader_msg_sptr;

stt_tag_reader_msg_sptr stt_make_tag_reader_msg(size_t itemsize, gr_msg_queue_sptr msgq);

/*!
 * \brief output[i] = input[i]
 * \ingroup misc_blk
 *
 * When enabled (default), this block copies its input to its output.
 * When disabled, this block drops its input on the floor.
 *
 */
class stt_tag_reader_msg : public gr_block
{
  size_t		d_itemsize;
  gr_msg_queue_sptr	d_msgq;

  friend stt_tag_reader_msg_sptr stt_make_tag_reader_msg(size_t itemsize, gr_msg_queue_sptr msgq);
  stt_tag_reader_msg(size_t itemsize, gr_msg_queue_sptr );

 public:

  bool check_topology(int ninputs, int noutputs);

  int general_work(int noutput_items,
		   gr_vector_int &ninput_items,
		   gr_vector_const_void_star &input_items,
		   gr_vector_void_star &output_items);
};

#endif
