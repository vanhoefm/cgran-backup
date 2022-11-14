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

#include "pr_rate_estimate.h"

#include <iostream>
#include <boost/format.hpp>
#include <gr_io_signature.h>
#include "stream_p.h"
#include <sys/time.h>

using namespace std;
using namespace boost;

pr_rate_estimate_sptr pr_make_rate_estimate(size_t item_size)
{
	return pr_rate_estimate_sptr(new pr_rate_estimate(item_size));
}

pr_rate_estimate::pr_rate_estimate(size_t item_size) :
	gr_sync_block("rate_estimate",
			 gr_make_io_signature(1, 1, item_size),
			 gr_make_io_signature(0, 0, 0)),
	d_count(0),
	d_rate(0.0)
{
	gettimeofday(&d_stamp, 0);
}

pr_rate_estimate::~pr_rate_estimate()
{
}

int pr_rate_estimate::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	unique_lock<mutex> lock(d_count_lock);
	d_count += noutput_items;

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

double pr_rate_estimate::rate()
{
	struct timeval t;
	struct timeval diff;

	gettimeofday(&t, 0);

	timersub(&t, &d_stamp, &diff);

	if(diff.tv_sec > 0)
	{
		double secs;
		unique_lock<mutex> lock(d_count_lock);

		secs = (double)diff.tv_sec + (double)diff.tv_usec/1000000.0;
		d_rate = (double)d_count/secs;
		d_count = 0;
		d_stamp = t;
	}

	return d_rate;
}

void pr_rate_estimate::clear()
{
	unique_lock<mutex> lock(d_count_lock);

	gettimeofday(&d_stamp, 0);
	d_count = 0;
	d_rate = 0.0;
}

