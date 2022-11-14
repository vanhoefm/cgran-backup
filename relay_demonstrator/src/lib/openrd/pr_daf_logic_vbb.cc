/* -*- c++ -*- */
/*
 * Copyright 2011 Anton Blad.
 * 
 * This file is part of OpenRD
 * 
 * OpenRD is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * OpenRD is distributed in the hope that it will be useful,
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

#include "openrd_debug.h"

#include "pr_daf_logic_vbb.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"
#include <iostream>
#include <cstring>

pr_daf_logic_vbb_sptr pr_make_daf_logic_vbb(int block_size)
{
	return pr_daf_logic_vbb_sptr(new pr_daf_logic_vbb(block_size));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_daf_logic_vbb::pr_daf_logic_vbb(int block_size) :
	gr_block("daf_logic_vbb",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<rxmeta,char>::alloc_size(block_size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<txmeta,char>::alloc_size(block_size))),
	d_block_size(block_size),
	d_pkt(0),
	d_frame(0),
	d_data_valid(0)
{
}

pr_daf_logic_vbb::~pr_daf_logic_vbb()
{
}

int pr_daf_logic_vbb::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const rxmeta,const char> in(input_items[0], d_block_size, ninput_items[0]);
	stream_p<txmeta,char> out(output_items[0], d_block_size, noutput_items);

	while(!in.atend() && !out.atend())
	{

		d_pkt = in.meta().pkt_seq;
		out.meta().pkt_seq = d_pkt;
		out.meta().frame_seq = d_frame;
		d_data_valid = !in.meta().decoded;
		out.meta().data_valid = d_data_valid;

		std::memcpy(out.data(), in.data(), d_block_size*sizeof(char));

		in.next();
		out.next();
	}

	consume_each(in.current());

	work_used(this, 0, in.current());
	work_exit(this, out.current());

	return out.current();
}

