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

#include "pr_const_mapper_vbc.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"
#include <iostream>
#include <cstring>
#include <stdio.h>

pr_const_mapper_vbc_sptr pr_make_const_mapper_vbc(int frame_size, modulation_type modulation)
{
	return pr_const_mapper_vbc_sptr(new pr_const_mapper_vbc(frame_size, modulation));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_const_mapper_vbc::pr_const_mapper_vbc(int frame_size, modulation_type modulation) :
	gr_block("const_mapper_vbc",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<txmeta,char>::alloc_size(frame_size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<txmeta,gr_complex>::alloc_size(frame_size))),
	d_frame_size(frame_size),
	d_pkt(0),
	d_data_valid(0),
	d_modulation(modulation)
{
}

pr_const_mapper_vbc::~pr_const_mapper_vbc()
{
}

int pr_const_mapper_vbc::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const txmeta,const char> in(input_items[0], d_frame_size, ninput_items[0]);
	stream_p<nullmeta,gr_complex> out(output_items[0], d_frame_size, noutput_items);


	while(!in.atend() && !out.atend())
	{
		d_data_valid = in.meta().data_valid;

		if(d_data_valid)
		{

			switch(d_modulation)
			{
			case MODULATION_BPSK:

				for(int k = 0; k < d_frame_size; k++)
				{
					if(in.data()[k] == 0)
						out.data()[k] = -1;
					else
						out.data()[k] = 1;			
				}
				break;

			case MODULATION_QPSK:
				for(int k = 0; k < d_frame_size; k++)
				{}
				break;
			}		
		}

		else
		{	

			for(int k = 0; k < d_frame_size; k++)
			{
				out.data()[k] = 0;			
			}		
		}

		in.next();
		out.next();
	}
	

	consume_each(in.current());

	work_used(this, 0, in.current());
	work_exit(this, out.current());

	return out.current();
}

modulation_type pr_const_mapper_vbc::modulation() const
{
	return d_modulation;
}

