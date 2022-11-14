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

#include "pr_pvec_extract.h"

#include <gr_io_signature.h>
#include "pvec.h"

pr_pvec_extract_sptr pr_make_pvec_extract(int insize, int offset, int outsize)
{
	return pr_pvec_extract_sptr(new pr_pvec_extract(insize, offset, outsize));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_pvec_extract::pr_pvec_extract(int insize, int offset, int outsize) :
	gr_sync_block("pvec_extract",
			 gr_make_io_signature(MIN_IN, MAX_IN, pvec_alloc_size(insize)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, pvec_alloc_size(outsize))),
	d_insize(insize),
	d_offset(offset),
	d_outsize(outsize)
{
}

pr_pvec_extract::~pr_pvec_extract()
{
}

int pr_pvec_extract::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	unsigned char* in = (unsigned char*) input_items[0];
	unsigned char* out = (unsigned char*) output_items[0];
	int inblock = input_signature()->sizeof_stream_item(0);
	int outblock = output_signature()->sizeof_stream_item(0);

	for(int nout = 0; nout < noutput_items; nout++)
	{
		std::memcpy(out, in+d_offset, d_outsize);
		pvec_pad(out, outblock, d_outsize);
		out += outblock;
		in += inblock;
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

int pr_pvec_extract::insize() const
{
	return d_insize;
}

int pr_pvec_extract::offset() const
{
	return d_offset;
}

int pr_pvec_extract::outsize() const
{
	return d_outsize;
}

