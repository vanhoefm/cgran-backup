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

#include "pr_pvec_concat.h"

#include <gr_io_signature.h>
#include <algorithm>
#include <numeric>
#include "pvec.h"
#include <cstring>

using namespace std;

pr_pvec_concat_sptr pr_make_pvec_concat(const vector<int>& sizes)
{
	return pr_pvec_concat_sptr(new pr_pvec_concat(sizes));
}

pr_pvec_concat::pr_pvec_concat(const vector<int>& sizes) :
	gr_sync_block("pvec_concat",
			 gr_make_io_signature(0, 0, 0),
			 gr_make_io_signature(0, 0, 0))
{
	int numin = sizes.size();
	vector<int> insizes;

	d_outsize = accumulate(sizes.begin(), sizes.end(), 0);

	insizes.resize(sizes.size());
	transform(sizes.begin(), sizes.end(), insizes.begin(), pvec_alloc_size);

	set_input_signature(gr_make_io_signaturev(numin, numin, insizes));
	set_output_signature(gr_make_io_signature(1, 1, pvec_alloc_size(d_outsize)));

	d_sizes = sizes;
}

pr_pvec_concat::~pr_pvec_concat()
{
}

int pr_pvec_concat::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	unsigned char* out = (unsigned char*)output_items[0];
	int outblock = output_signature()->sizeof_stream_item(0);

	for(int nout = 0; nout < noutput_items; nout++)
	{
		int idx = 0;

		for(unsigned int nin = 0; nin < d_sizes.size(); nin++)
		{
			memcpy(&out[idx], input_items[nin], d_sizes[nin]);
			input_items[nin] = (char*)input_items[nin] + input_signature()->sizeof_stream_item(nin);
			idx += d_sizes[nin];
		}

		pvec_pad(out, outblock, d_outsize);
		out += outblock;
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);

	return noutput_items;
}

const vector<int>& pr_pvec_concat::sizes() const
{
	return d_sizes;
}

int pr_pvec_concat::outsize() const
{
	return d_outsize;
}

