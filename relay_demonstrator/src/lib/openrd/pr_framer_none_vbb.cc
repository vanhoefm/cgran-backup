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

#include "pr_framer_none_vbb.h"

#include <gr_io_signature.h>
#include "stream_p.h"
#include <cstring>
#include <algorithm>

pr_framer_none_vbb_sptr pr_make_framer_none_vbb(int frame_size)
{
	return pr_framer_none_vbb_sptr(new pr_framer_none_vbb(frame_size));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_framer_none_vbb::pr_framer_none_vbb(int frame_size) :
	pr_framer_vbb(frame_size, frame_size)
{
}

pr_framer_none_vbb::~pr_framer_none_vbb()
{
}

int pr_framer_none_vbb::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const txmeta,const char> in(input_items[0], frame_size(), ninput_items[0]);
	stream_p<txmeta,char> out(output_items[0], frame_size(), noutput_items);

	while(!in.atend() && !out.atend())
	{
		out.meta().pkt_seq = in.meta().pkt_seq;
		out.meta().frame_seq = in.meta().frame_seq;
		out.meta().data_valid = in.meta().data_valid;
		std::copy(in.data(), in.data()+frame_size(), out.data());
		in.next();
		out.next();
	}

	consume(0, in.current());

	work_used(this, 0, in.current());
	work_exit(this, out.current());

	return out.current();
}

