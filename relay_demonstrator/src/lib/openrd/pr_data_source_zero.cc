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

#include "pr_data_source_zero.h"

#include <gr_io_signature.h>
#include <cstring>

pr_data_source_zero_sptr pr_make_data_source_zero(int block_size)
{
	return pr_data_source_zero_sptr(new pr_data_source_zero(block_size));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_data_source_zero::pr_data_source_zero(int block_size) :
	pr_data_source(block_size)
{
}

pr_data_source_zero::~pr_data_source_zero()
{
}

void pr_data_source_zero::fill_packet(unsigned char* data, unsigned char* valid)
{
	std::memset(data, 0, block_size());
	*valid = 1;
}

