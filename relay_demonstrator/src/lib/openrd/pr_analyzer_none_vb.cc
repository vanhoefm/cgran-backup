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

#include "pr_analyzer_none_vb.h"

#include <gr_io_signature.h>
#include "stream_p.h"

pr_analyzer_none_vb_sptr pr_make_analyzer_none_vb(int block_size)
{
	return pr_analyzer_none_vb_sptr(new pr_analyzer_none_vb(block_size));
}

static const int MIN_IN = 2;
static const int MAX_IN = 2;
static const int MIN_OUT = 0;
static const int MAX_OUT = 0;

pr_analyzer_none_vb::pr_analyzer_none_vb(int block_size) :
	pr_analyzer_vb(block_size)
{
}

pr_analyzer_none_vb::~pr_analyzer_none_vb()
{
}

void pr_analyzer_none_vb::analyze(const txmeta& refmeta, const char* ref,
		const rxmeta& recmeta, const char* rec)
{
}

