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

#include "pr_analyzer_ber_vb.h"
#include "pvec.h"

#include <gr_io_signature.h>
#include <vector>
#include "stream_p.h"
#include <iostream>
#include <boost/format.hpp>

using namespace std;
using namespace boost;

pr_analyzer_ber_vb_sptr pr_make_analyzer_ber_vb(int block_size, int avg_blocks)
{
	return pr_analyzer_ber_vb_sptr(new pr_analyzer_ber_vb(block_size, avg_blocks));
}

pr_analyzer_ber_vb::pr_analyzer_ber_vb(int block_size, int avg_blocks) :
	pr_analyzer_vb(block_size),
	d_block_size(block_size),
	d_avg_blocks(avg_blocks),
	d_bits(0),
	d_correctbits(0),
	d_packets(0),
	d_correctpackets(0)
{
}

pr_analyzer_ber_vb::~pr_analyzer_ber_vb()
{
}

void pr_analyzer_ber_vb::analyze(const txmeta& refmeta, const char* ref,
		const rxmeta& recmeta, const char* rec)
{
	int nright = 0;

	if(!refmeta.data_valid)
		return;

	// Compare the bits in the current block
	for(int k = 0; k < d_block_size; k++)
		if(ref[k] == rec[k])
			nright++;

	d_bits += d_block_size;
	d_correctbits += nright;
	d_packets++;
	if(nright == d_block_size)
		d_correctpackets++;

	// If avg_blocks is reached, produce measurement outputs
	if(d_packets == d_avg_blocks)
	{
		float avg_ber = (double)(d_bits-d_correctbits)/(double)d_bits;
		float avg_bler = (double)(d_packets-d_correctpackets)/(double)d_packets;

		boost::unique_lock<boost::mutex> lock(d_ber_lock);

		d_ber.push_back(avg_ber);
		d_bler.push_back(avg_bler);

		d_bits = 0;
		d_correctbits = 0;
		d_packets = 0;
		d_correctpackets = 0;
	}
}

std::vector<float> pr_analyzer_ber_vb::ber()
{
	boost::unique_lock<boost::mutex> lock(d_ber_lock);

	std::vector<float> tmp1 = d_ber;
	d_ber.clear();
	return tmp1;
}

std::vector<float> pr_analyzer_ber_vb::bler()
{
	boost::unique_lock<boost::mutex> lock(d_ber_lock);

	std::vector<float> tmp2 = d_bler;
	d_bler.clear();
	return tmp2;
}



