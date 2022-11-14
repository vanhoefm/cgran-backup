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

#include <pr_frame_sync_cc.h>
#include <gr_io_signature.h>
#include <cstring>
#include "stream_p.h"
#include "frametime.h"

pr_frame_sync_cc_sptr pr_make_frame_sync_cc(int frame_size)
{
	return pr_frame_sync_cc_sptr(new pr_frame_sync_cc(frame_size));
}

pr_frame_sync_cc::pr_frame_sync_cc(int frame_size) :
	gr_block("frame_sync_cc",
		gr_make_io_signature2(2, 2, sizeof(gr_complex), sizeof(unsigned int)),
		gr_make_io_signature(1, 1, stream_p<rxframe,gr_complex>::alloc_size(frame_size))),
	d_state(STATE_NOSYNC),
	d_frame_type(0),
	d_frame_size(frame_size),
	d_cnt(0),
	d_frame(new gr_complex[d_frame_size])
{
	set_relative_rate(1.0/(double)frame_size);
}


pr_frame_sync_cc::~pr_frame_sync_cc()
{
	delete[] d_frame;
}

int pr_frame_sync_cc::general_work(int noutput_items, gr_vector_int& ninput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const gr_complex* in = (const gr_complex*)input_items[0];
	const unsigned int* sync = (const unsigned int*)input_items[1];
	stream_p<rxframe, gr_complex> out(output_items[0], d_frame_size, noutput_items);
	int nin = 0;

	while(nin < ninput_items[0] && nin < ninput_items[1] && !out.atend())
	{
		if(d_state == STATE_NOSYNC)
		{
			if(sync[nin] > 0)
			{
				// Start of frame
				d_frame_type = sync[nin];
				d_cnt = 0;
				d_state = STATE_SYNC;
			}
		}

		if(d_state == STATE_SYNC)
		{
			d_frame[d_cnt] = in[nin];
			d_cnt++;
			if(d_cnt == d_frame_size)
			{
				// End of frame, fill in output element
				out.meta().power = 0;
				out.meta().stamp = frametime_now();
				out.meta().frame_type = d_frame_type;
				std::memcpy(out.data(), d_frame, d_frame_size*sizeof(gr_complex));
				out.next();
				d_state = STATE_NOSYNC;
			}
		}
		nin++;
	}

	work_used(this, 0, nin);
	work_used(this, 1, nin);
	work_exit(this, out.current());

	consume_each(nin);

	return out.current();
}

int pr_frame_sync_cc::frame_size() const
{
	return d_frame_size;
}

