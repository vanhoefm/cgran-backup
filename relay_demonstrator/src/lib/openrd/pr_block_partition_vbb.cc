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

#include "pr_block_partition_vbb.h"

#include <gr_io_signature.h>
#include <gr_complex.h>
#include "pvec.h"
#include "stream_p.h"
#include <algorithm>

pr_block_partition_vbb_sptr pr_make_block_partition_vbb(int insize, int outsize)
{
	return pr_block_partition_vbb_sptr(new pr_block_partition_vbb(insize, outsize));
}

pr_block_partition_vbb::pr_block_partition_vbb(int insize, int outsize) :
	gr_sync_interpolator("pr_block_partition_vbb",
			gr_make_io_signature(1, 1, stream_p<txmeta,char>::alloc_size(insize)),
			gr_make_io_signature(1, 1, stream_p<txmeta,char>::alloc_size(outsize)), 
			1),
	d_insize(insize),
	d_outsize(outsize)
{
	if(insize%outsize != 0)
	{
		throw "pr_block_partition_vbb: insize not a multiple of outsize";
	}

	d_factor = insize/outsize;
	set_interpolation(d_factor);
}

pr_block_partition_vbb::~pr_block_partition_vbb()
{
}

int pr_block_partition_vbb::work(int noutput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<const txmeta,const char> in(input_items[0], d_insize, noutput_items/d_factor);
	stream_p<txmeta,char> out(output_items[0], d_outsize, noutput_items);

	while(!out.atend())
	{
		for(int b = 0; b < d_factor; b++)
		{
			out.meta().pkt_seq = in.meta().pkt_seq;
			out.meta().frame_seq = b;
			out.meta().data_valid = in.meta().data_valid;
			std::copy(in.data() + b*d_outsize, in.data() + (b+1)*d_outsize, out.data());
			out.next();
		}
		in.next();
	}

	work_used(this, 0, out.current());
	work_exit(this, out.current());

	return out.current();
}

