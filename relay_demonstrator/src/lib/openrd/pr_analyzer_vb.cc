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

#include "pr_analyzer_vb.h"
#include "stream_p.h"

#include <gr_io_signature.h>
#include <stdexcept>
#include <iostream>
#include <boost/format.hpp>

using namespace std;
using namespace boost;

static const std::vector<int> input_streams(int s)
{
	std::vector<int> v;

	v.push_back(stream_p<txmeta,char>::alloc_size(s));
	v.push_back(stream_p<rxmeta,char>::alloc_size(s));

	return v;
}

pr_analyzer_vb::pr_analyzer_vb(int block_size) :
	gr_sync_block("analyzer_vb",
			gr_make_io_signaturev(2, 2, input_streams(block_size)),
			gr_make_io_signature(0, 0, 0)),	
	d_block_size(block_size)		
{
}

pr_analyzer_vb::pr_analyzer_vb(int block_size, gr_io_signature_sptr out) :
	gr_sync_block("analyzer_vb",
			gr_make_io_signaturev(2, 2, input_streams(block_size)),
			out),	
	d_block_size(block_size)		
{
}

pr_analyzer_vb::~pr_analyzer_vb()
{
}

int pr_analyzer_vb::block_size() const
{
	return d_block_size;
}

int pr_analyzer_vb::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<const txmeta,const char> in0(input_items[0], d_block_size, noutput_items);
	const_stream_p<const rxmeta,const char> in1(input_items[1], d_block_size, noutput_items);

	while(!in0.atend())
	{
		analyze(in0.meta(), in0.data(), in1.meta(), in1.data());

		in0.next();
		in1.next();
	}

	work_used(this, 0, noutput_items);
	work_used(this, 1, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

