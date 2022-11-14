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

#include "pr_deframer_vbb.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"
#include "snr.h"
#include <iostream>
#include <boost/format.hpp>

using namespace boost;
using namespace std;

pr_deframer_vbb::pr_deframer_vbb(int frame_size, int data_size) :
	gr_block("deframer_vbb",
			 gr_make_io_signature(1, 1, stream_p<rxframe,char>::alloc_size(frame_size)),
			 gr_make_io_signature(1, 1, stream_p<rxframe,char>::alloc_size(data_size))),
	d_frame_size(frame_size),
	d_data_size(data_size),
	d_storesnr(false),
	d_autosnr(true)
{
}

pr_deframer_vbb::~pr_deframer_vbb()
{
}

int pr_deframer_vbb::frame_size() const
{
	return d_frame_size;
}

int pr_deframer_vbb::data_size() const
{
	return d_data_size;
}

void pr_deframer_vbb::set_storesnr(bool storesnr)
{
	d_storesnr = storesnr;
}

void pr_deframer_vbb::set_autosnr(bool autosnr)
{
	d_autosnr = autosnr;
}

vector<double> pr_deframer_vbb::snr()
{
	vector<double> v;

	unique_lock<mutex> lock(d_snr_lock);

	v = d_snr;
	d_snr.clear();

	return v;
}

int pr_deframer_vbb::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const rxframe, const char> in(input_items[0], d_frame_size, ninput_items[0]);
	stream_p<rxframe, char> out(output_items[0], d_data_size, noutput_items);

	while(!in.atend() && !out.atend())
	{
		bool advance;

		advance = deframe(in.meta(), in.data(), out.meta(), out.data());

		if(advance)
		{
			out.meta().stamp = in.meta().stamp;

			if(d_autosnr)
				out.meta().power = 0; //snr_estimate(in.data(), d_frame_size);

			if(d_storesnr)
			{
				unique_lock<mutex> lock(d_snr_lock);
				d_snr.push_back(out.meta().power);
			}

			out.next();
		}

		in.next();
	}

	consume_each(in.current());

	work_used(this, 0, in.current());
	work_exit(this, out.current());

	return out.current();
}

