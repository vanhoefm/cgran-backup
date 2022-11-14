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

#include "pr_insert_head.h"

#include <gr_io_signature.h>
#include <cstring>

pr_insert_head_sptr pr_make_insert_head(int sizeof_stream_item, int nitems)
{
	return pr_insert_head_sptr(new pr_insert_head(sizeof_stream_item, nitems));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_insert_head::pr_insert_head(int sizeof_stream_item, int nitems) :
	gr_block("insert_head",
			 gr_make_io_signature(MIN_IN, MAX_IN, sizeof_stream_item),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, sizeof_stream_item)),
	d_sizeof_stream_item(sizeof_stream_item),
	d_nitems(nitems),
	d_current(0)
{
}

pr_insert_head::~pr_insert_head()
{
}

int pr_insert_head::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	int nin;
	int nout;

	if(d_current < d_nitems)
	{
		nout = std::min(d_nitems-d_current, noutput_items);
		std::memset(output_items[0], 0, nout*d_sizeof_stream_item);
		d_current += nout;
		nin = 0;
	}
	else
	{
		nout = std::min(noutput_items, ninput_items[0]);
		std::memcpy(output_items[0], input_items[0], nout*d_sizeof_stream_item);
		nin = nout;
	}

	consume(0, nin);

	work_used(this, 0, nin);
	work_exit(this, nout);

	return nout;
}

void pr_insert_head::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
	if(d_current < d_nitems)
	{
		ninput_items_required[0] = 0;
	}
	else
	{
		ninput_items_required[0] = noutput_items;
	}
}

