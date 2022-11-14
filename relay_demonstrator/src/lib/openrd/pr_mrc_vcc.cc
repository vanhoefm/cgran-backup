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

#include "pr_mrc_vcc.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"
#include <gr_complex.h>

pr_mrc_vcc_sptr pr_make_mrc_vcc(int size)
{
	return pr_mrc_vcc_sptr(new pr_mrc_vcc(size));
}

static const int MIN_IN = 2;
static const int MAX_IN = 2;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_mrc_vcc::pr_mrc_vcc(int size) :
	gr_sync_block("mrc_vcc",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<rxframe,gr_complex>::alloc_size(size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<rxframe,gr_complex>::alloc_size(size))),
	d_size(size)
{
}

pr_mrc_vcc::~pr_mrc_vcc()
{
}

int pr_mrc_vcc::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<const rxframe,const gr_complex> in0(input_items[0], d_size, noutput_items);
	const_stream_p<const rxframe,const gr_complex> in1(input_items[1], d_size, noutput_items);
	stream_p<rxframe,gr_complex> out(output_items[0], d_size, noutput_items);

	while(!out.atend())
	{
		float scale0 = sqrt(in0.meta().power);
		float scale1 = sqrt(in1.meta().power);

		if(scale0 == 0.0 && scale1 == 0.0)
		{
			scale0 = 1.0;
			scale1 = 0.0;
		}
		else if(scale0 > 10.0*scale1)
		{
			scale0 = 1.0;
			scale1 = 0.0;
		}
		else if(scale1 > 10.0*scale0)
		{
			scale0 = 0.0;
			scale1 = 1.0;
		}

		out.meta().power = in0.meta().power + in1.meta().power;
		out.meta().pkt_seq = in0.meta().pkt_seq;
		out.meta().frame_seq = in0.meta().frame_seq;

		for(int k = 0; k < d_size; k++)
			out.data()[k] = scale0*in0.data()[k] + scale1*in1.data()[k];

		in0.next();
		in1.next();
		out.next();
	}

	work_used(this, 0, out.current());
	work_used(this, 1, out.current());
	work_exit(this, out.current());

	return out.current();
}

int pr_mrc_vcc::size() const
{
	return d_size;
}

