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

#include "pr_ber_estimate_b.h"

#include <iostream>
#include <boost/format.hpp>
#include <gr_io_signature.h>
#include "stream_p.h"

using namespace std;
using namespace boost;

pr_ber_estimate_b_sptr pr_make_ber_estimate_b(double alpha)
{
	return pr_ber_estimate_b_sptr(new pr_ber_estimate_b(alpha));
}

pr_ber_estimate_b::pr_ber_estimate_b(double alpha) :
	gr_sync_block("ber_estimate_b",
			 gr_make_io_signature(2, 2, sizeof(char)),
			 gr_make_io_signature(0, 0, 0)),
	d_ber(0)
{
	set_alpha(alpha);
}

pr_ber_estimate_b::~pr_ber_estimate_b()
{
}

int pr_ber_estimate_b::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const char* in0 = (const char*)input_items[0];
	const char* in1 = (const char*)input_items[1];

	unique_lock<mutex> lock(d_ber_lock);

	for(int n = 0; n < noutput_items; n++)
	{
		d_ber *= d_beta;

		if(in0[n] != in1[n])
			d_ber += d_alpha;
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

void pr_ber_estimate_b::set_alpha(double alpha)
{
	d_alpha = alpha;
	d_beta = 1-alpha;
}

double pr_ber_estimate_b::ber() const
{
	unique_lock<mutex> lock(d_ber_lock);

	return d_ber;
}

void pr_ber_estimate_b::clear()
{
	unique_lock<mutex> lock(d_ber_lock);

	d_ber = 0;
}

