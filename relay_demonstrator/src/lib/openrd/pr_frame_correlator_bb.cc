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

#include "pr_frame_correlator_bb.h"

#include <gr_io_signature.h>
#include "pvec.h"

#include <iostream>

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_frame_correlator_bb::pr_frame_correlator_bb(int input_size, int frame_size) :
	gr_sync_block("frame_correlator_bb",
			 gr_make_io_signature(MIN_IN, MAX_IN, pvec_alloc_size(input_size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, 4)),
	d_input_size(input_size),
	d_frame_size(frame_size)
{
	set_history(delay());
}

pr_frame_correlator_bb::~pr_frame_correlator_bb()
{
}

int pr_frame_correlator_bb::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const char* in = (const char*) input_items[0];
	unsigned int* out = (unsigned int*) output_items[0];
	int inblock = input_signature()->sizeof_stream_item(0);

	for(int k = 0; k < noutput_items; k++)
	{
		out[k] = correlate(in);
		in += inblock;
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

int pr_frame_correlator_bb::frame_size() const
{
	return d_frame_size;
}

int pr_frame_correlator_bb::input_size() const
{
	return d_input_size;
}

int pr_frame_correlator_bb::delay() const
{
	return d_frame_size/d_input_size;
}


