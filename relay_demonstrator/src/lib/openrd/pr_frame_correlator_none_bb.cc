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

#include "pr_frame_correlator_none_bb.h"

#include <gr_io_signature.h>

pr_frame_correlator_none_bb_sptr pr_make_frame_correlator_none_bb(int input_size, int frame_size)
{
	return pr_frame_correlator_none_bb_sptr(new pr_frame_correlator_none_bb(input_size, frame_size));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_frame_correlator_none_bb::pr_frame_correlator_none_bb(int input_size, int frame_size) :
	pr_frame_correlator_bb(input_size, frame_size),
	d_cnt(0)
{
}

pr_frame_correlator_none_bb::~pr_frame_correlator_none_bb()
{
}

int pr_frame_correlator_none_bb::correlate(const char* data)
{
	int corr = 0;

	if(d_cnt == delay()-1)
	{
		d_cnt = 0;
		corr = 1;
	}
	else
		d_cnt++;

	return corr;
}

