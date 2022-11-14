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

#include "pr_framer_vbb.h"

#include <gr_io_signature.h>
#include "pvec.h"
#include "stream_p.h"

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_framer_vbb::pr_framer_vbb(int data_size, int frame_size) :
	gr_block("framer_vbb",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<txmeta,char>::alloc_size(data_size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<txmeta,char>::alloc_size(frame_size*sizeof(char)))),
	d_data_size(data_size), d_frame_size(frame_size)
{
}

pr_framer_vbb::~pr_framer_vbb()
{
}

int pr_framer_vbb::data_size() const
{
	return d_data_size;
}

int pr_framer_vbb::frame_size() const
{
	return d_frame_size;
}

