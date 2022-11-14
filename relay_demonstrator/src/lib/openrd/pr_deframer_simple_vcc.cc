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

#include "pr_deframer_simple_vcc.h"

#include <gr_io_signature.h>
#include "stream_p.h"
#include <cstring>
#include <gr_complex.h>
#include <algorithm>
#include <iostream>
#include <boost/format.hpp>

using namespace std;
using namespace boost;

pr_deframer_simple_vcc_sptr pr_make_deframer_simple_vcc(int frame_size, const std::vector<char>& access_code)
{
	return pr_deframer_simple_vcc_sptr(new pr_deframer_simple_vcc(frame_size, access_code));
}

pr_deframer_simple_vcc::pr_deframer_simple_vcc(int frame_size, const std::vector<char>& access_code) :
	pr_deframer_vcc(frame_size, frame_size-access_code.size()), d_pkt(0), d_access_code(access_code)
{
}

pr_deframer_simple_vcc::~pr_deframer_simple_vcc()
{
}

bool pr_deframer_simple_vcc::deframe(const rxframe& inmeta, const gr_complex* in,
		rxframe& outmeta, gr_complex* out)
{
	if(inmeta.frame_type != 1)
	{
		cerr << format("pr_deframer_simple_cc: Invalid frame type %d") % inmeta.frame_type << endl;
		return false;
	}

	outmeta.pkt_seq = d_pkt;
	outmeta.frame_seq = 0;
	d_pkt++;

	copy(in+d_access_code.size(), in+frame_size(), out);

	return true;
}

