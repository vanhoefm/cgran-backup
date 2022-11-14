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

#include "pr_block_coder_vbb.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_block_coder_vbb::pr_block_coder_vbb(int information_size, int codeword_size) :
	gr_sync_block("block_coder_vbb",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<txmeta,char>::alloc_size(information_size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<txmeta,char>::alloc_size(codeword_size))),
	d_information_size(information_size),
	d_codeword_size(codeword_size)
{
}

pr_block_coder_vbb::~pr_block_coder_vbb()
{
}

int pr_block_coder_vbb::information_size() const
{
	return d_information_size;
}

int pr_block_coder_vbb::codeword_size() const
{
	return d_codeword_size;
}

int pr_block_coder_vbb::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<const txmeta,const char> in(input_items[0], d_information_size, noutput_items);
	stream_p<txmeta,char> out(output_items[0], d_codeword_size, noutput_items);

	while(!out.atend())
	{
		out.meta().pkt_seq = in.meta().pkt_seq;
		out.meta().frame_seq = in.meta().frame_seq;
		out.meta().data_valid = in.meta().data_valid;
		encode(out.data(), in.data());
		
		in.next();
		out.next();
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;

}

