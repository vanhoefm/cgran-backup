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

#include "%lib_%class.h"

#include <iostream>
#include <boost/format.hpp>
#include <gr_io_signature.h>
#include "stream_p.h"

using namespace std;
using namespace boost;

%lib_%class_sptr %lib_make_%class(%args)
{
	return %lib_%class_sptr(new %lib_%class(%argb));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

%lib_%class::%lib_%class(%args) :
	gr_sync_block("%class",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<>::alloc_size()),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<>::alloc_size()))
{
}

%lib_%class::~%lib_%class()
{
}

int %lib_%class::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<> in0(input_items[0], size, noutput_items);
	stream_p<> out0(output_items[0], size, noutput_items);

	work_used(this, 0, out0.current());
	work_exit(this, out0.current());

	return out0.current();
}

