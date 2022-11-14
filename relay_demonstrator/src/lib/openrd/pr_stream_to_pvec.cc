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

#include "pr_stream_to_pvec.h"

#include <gr_io_signature.h>
#include <cstring>
#include "pvec.h"

pr_stream_to_pvec_sptr pr_make_stream_to_pvec(int item_size, int nitems)
{
	return pr_stream_to_pvec_sptr(new pr_stream_to_pvec(item_size, nitems));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_stream_to_pvec::pr_stream_to_pvec(int item_size, int nitems) :
	gr_sync_decimator("stream_to_pvec",
			 gr_make_io_signature(MIN_IN, MAX_IN, item_size),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, pvec_alloc_size(item_size*nitems)),
			 nitems),
	d_item_size(item_size), 
	d_nitems(nitems),
	d_alloc_size(pvec_alloc_size(item_size*nitems)),
	d_block_size(item_size*nitems)
{
	d_pad_size = output_signature()->sizeof_stream_item(0) - d_block_size;
}

pr_stream_to_pvec::~pr_stream_to_pvec()
{
}

int pr_stream_to_pvec::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const char* in = (const char*) input_items[0];
	char* out = (char*) output_items[0];

	int nout = 0;

	while(nout < noutput_items)
	{
		std::memcpy(out, in, d_block_size);
		std::memset(out+d_block_size, 0, d_pad_size);
		out += d_block_size+d_pad_size;
		in += d_block_size;
		nout++;
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

int pr_stream_to_pvec::item_size() const
{
	return d_item_size;
}

int pr_stream_to_pvec::nitems() const
{
	return d_nitems;
}

int pr_stream_to_pvec::alloc_size() const
{
	return d_alloc_size;
}

