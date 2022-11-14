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

#include "pr_frame_correlator_simple_bb.h"

#include <gr_io_signature.h>

pr_frame_correlator_simple_bb_sptr pr_make_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired)
{
	return pr_frame_correlator_simple_bb_sptr(new pr_frame_correlator_simple_bb(input_size, frame_size, access_code, nrequired));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_frame_correlator_simple_bb::pr_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired) :
	pr_frame_correlator_bb(input_size, frame_size),
	d_cnt(0),
	d_access_code(access_code), 
	d_nrequired(nrequired)
{
}

pr_frame_correlator_simple_bb::~pr_frame_correlator_simple_bb()
{
}

int pr_frame_correlator_simple_bb::correlate(const char* data)
{
	int corr = 0;

	for(unsigned int i = 0; i < d_access_code.size(); i++)
		corr += (data[i] == d_access_code[i]);

	if(corr >= d_nrequired)
		return 1;
	else
		return 0;
}

