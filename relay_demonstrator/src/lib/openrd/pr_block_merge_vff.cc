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

#include "pr_block_merge_vff.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"
#include <iostream>
#include <cstring>

pr_block_merge_vff_sptr pr_make_block_merge_vff(int insize, int outsize)
{
	return pr_block_merge_vff_sptr(new pr_block_merge_vff(insize, outsize));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_block_merge_vff::pr_block_merge_vff(int insize, int outsize) :
	gr_block("block_merge_vff",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<rxframe,float>::alloc_size(insize)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<rxmeta,float>::alloc_size(outsize))),
	d_insize(insize),
	d_outsize(outsize),
	d_parts(outsize/insize),
	d_pkt(0),
	d_first_part(true),
	d_block(new float[outsize]),
	d_numpktseq(2048)
{
	if(outsize%insize != 0)
	{
		throw "pr_block_merge_vbb: outsize not a multiple of insize";
	}

	set_relative_rate(1.0/d_parts);
}

pr_block_merge_vff::~pr_block_merge_vff()
{
	delete[] d_block;
}

int pr_block_merge_vff::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const rxframe,const float> in(input_items[0], d_insize, ninput_items[0]);
	stream_p<rxmeta,float> out(output_items[0], d_outsize, noutput_items);

	while(!in.atend() && !out.atend())
	{
		// If first partition of the frame, set packet sequence number and 
		// initialize data to zeroes.
		if(d_first_part)
		{
			d_first_part = false;
			std::memset(d_block, 0, d_outsize*sizeof(float));
		}

		// Check that packet sequence number is legal
		if(in.meta().pkt_seq >= d_numpktseq)
		{
			std::cerr << "pr_block_merge_vff: input frame has illegal pkt_seq " << in.meta().pkt_seq << ", ignoring" << std::endl;
			in.next();
			continue;
		}

		// If this is a new packet, copy the buffer and advance the output stream.
		if(d_pkt != in.meta().pkt_seq)
		{
			out.meta().pkt_seq = d_pkt;
			std::memcpy(out.data(), d_block, d_outsize*sizeof(float));
			out.next();
			d_pkt++;
			if(d_pkt == d_numpktseq)
				d_pkt = 0;
			d_first_part = true;
			continue;
		}

		// Copy the partition data if frame sequence number is valid.
		if(in.meta().frame_seq < d_parts)
		{
			std::memcpy(d_block + d_insize*in.meta().frame_seq, in.data(), d_insize*sizeof(float));
		}
		else
		{
			std::cerr << "pr_block_merge_vff: input frame has illegal frame_seq " << in.meta().frame_seq << std::endl;
		}

		in.next();
	}

	consume_each(in.current());

	work_used(this, 0, in.current());
	work_exit(this, out.current());

	return out.current();
}

void pr_block_merge_vff::set_numpktseq(unsigned int numpktseq)
{
	d_numpktseq = numpktseq;
}

