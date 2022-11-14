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

#include "pr_data_sink.h"

#include <iostream>
#include <boost/format.hpp>
#include <gr_io_signature.h>
#include "stream_p.h"

using namespace std;
using namespace boost;

pr_data_sink::pr_data_sink(int block_size) :
	gr_sync_block("data_sink",
			 gr_make_io_signature(1, 1, stream_p<rxmeta,unsigned char>::alloc_size(block_size)),
			 gr_make_io_signature(0, 0, 0)),
	d_block_size(block_size),
	d_numpktseq(2048),
	d_pkt_seq(0)
{
}

pr_data_sink::~pr_data_sink()
{
}

int pr_data_sink::block_size() const
{
	return d_block_size;
}

void pr_data_sink::set_numpktseq(unsigned int numpktseq)
{
	d_numpktseq = numpktseq;
}

int pr_data_sink::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<const rxmeta,const unsigned char> in(input_items[0], d_block_size, noutput_items);

	while(!in.atend())
	{
		if(d_pkt_seq != in.meta().pkt_seq)
		{
			cerr << format("pr_data_sink: received seq %d, expected %d") % 
				in.meta().pkt_seq % d_pkt_seq << endl;

			d_pkt_seq = in.meta().pkt_seq;
		}

		handle_packet(in.data());

		d_pkt_seq++;
		if(d_pkt_seq >= d_numpktseq)
			d_pkt_seq -= d_numpktseq;

		in.next();
	}

	work_used(this, 0, in.current());
	work_exit(this, 0);

	return in.current();
}

