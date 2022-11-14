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

#include "pr_data_source.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"

static const int MIN_IN = 0;
static const int MAX_IN = 0;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_data_source::pr_data_source(int block_size) :
	gr_sync_block("data_source",
			 gr_make_io_signature(MIN_IN, MAX_IN, 0),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<txmeta,unsigned char>::alloc_size(block_size))),
	d_block_size(block_size),
	d_numpktseq(2048),
	d_pkt_seq(0)
{
}

pr_data_source::~pr_data_source()
{
}

int pr_data_source::block_size() const
{
	return d_block_size;
}

void pr_data_source::set_numpktseq(int numpktseq)
{
	d_numpktseq = numpktseq;
}

int pr_data_source::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	stream_p<txmeta,unsigned char> out(output_items[0], d_block_size, noutput_items);
	
	while(!out.atend())
	{
		unsigned char valid;

		out.meta().pkt_seq = d_pkt_seq;
		out.meta().frame_seq = 0;

		fill_packet(out.data(), &valid);

		out.meta().data_valid = valid;

		d_pkt_seq++;
		if(d_pkt_seq >= d_numpktseq)
			d_pkt_seq -= d_numpktseq;

		out.next();
		if(!valid)
			break;
	}

	work_exit(this, out.current());

	return out.current();
}

