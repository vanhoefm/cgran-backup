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

#include "pr_snr_estimate_c.h"

#include <iostream>
#include <boost/format.hpp>
#include <gr_io_signature.h>
#include "stream_p.h"
#include "snr.h"

using namespace std;
using namespace boost;

pr_snr_estimate_c_sptr pr_make_snr_estimate_c(modulation_type modulation, int block_size)
{
	return pr_snr_estimate_c_sptr(new pr_snr_estimate_c(modulation, block_size));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 0;
static const int MAX_OUT = 0;

pr_snr_estimate_c::pr_snr_estimate_c(modulation_type modulation, int block_size) :
	gr_sync_block("snr_estimate_c",
			 gr_make_io_signature(MIN_IN, MAX_IN, sizeof(gr_complex)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, 0)),
	d_modulation(modulation),
	d_block_size(block_size)
{
	if(modulation != MODULATION_BPSK && modulation != MODULATION_QPSK)
		throw "pr_snr_estimate_c: Unsupported modulation";

	set_output_multiple(block_size);
}

pr_snr_estimate_c::~pr_snr_estimate_c()
{
}

int pr_snr_estimate_c::block_size() const
{
	return d_block_size;
}

std::vector<double> pr_snr_estimate_c::snr()
{
	vector<double> v;
	unique_lock<mutex> lock(d_snr_lock);
	v = d_snr;
	d_snr.clear();
	return v;
}

int pr_snr_estimate_c::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	int nout = 0;
	double snr;

	work_enter(this, noutput_items, 0, input_items, output_items);

	const gr_complex* in = (const gr_complex*) input_items[0];

	while(noutput_items - nout >= d_block_size)
	{
		snr = snr_estimate(in, d_block_size);
		unique_lock<mutex> lock(d_snr_lock);
		d_snr.push_back(snr);
		nout += d_block_size;
	}

	work_used(this, 0, out0.current());
	work_exit(this, out0.current());

	return nout;
}

