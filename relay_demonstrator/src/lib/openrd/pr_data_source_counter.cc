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

#include "pr_data_source_counter.h"

#include <gr_io_signature.h>
#include "stream_p.h"

pr_data_source_counter_sptr pr_make_data_source_counter(int block_size)
{
	return pr_data_source_counter_sptr(new pr_data_source_counter(block_size));
}

static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_data_source_counter::pr_data_source_counter(int block_size) :
	pr_data_source(block_size)
{
}

pr_data_source_counter::~pr_data_source_counter()
{
}

void pr_data_source_counter::fill_packet(unsigned char* data, unsigned char* valid)
{
	int k;

	for(k = 0; k <= block_size()/4-1; k++)
	{
		data[4*k+3] = k & 0x01;
		data[4*k+2] = (k & 0x02) >> 1;
		data[4*k+1] = (k & 0x04) >> 2;
		data[4*k+0] = (k & 0x08) >> 3;
	}
	for(k=4*k; k < block_size(); k++)
		data[k] = 0;

	*valid = 1;
}

