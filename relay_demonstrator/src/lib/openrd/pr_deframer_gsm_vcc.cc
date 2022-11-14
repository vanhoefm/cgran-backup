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

#include "pr_deframer_gsm_vcc.h"

#include <gr_io_signature.h>
#include "stream_p.h"
#include <cstring>
#include <gr_complex.h>
#include <algorithm>
#include <math.h>
#include "field_coder.h"
#include "modulation.h"
#include <iostream>
#include "snr.h"
#include "gsm_frame.h"
#include <boost/format.hpp>

using namespace std;
using namespace boost;

pr_deframer_gsm_vcc_sptr pr_make_deframer_gsm_vcc(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code)
{
	return pr_deframer_gsm_vcc_sptr(new pr_deframer_gsm_vcc(frame_size, pktseq_code, sync_code, data_code));
}

pr_deframer_gsm_vcc::pr_deframer_gsm_vcc(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code) :
	pr_deframer_vcc(frame_size, frame_size-data_code.size()), d_pktseq_code(pktseq_code),
	d_sync_code(sync_code), d_data_code(data_code), d_syncvalid(false), d_pktseq(0), 
	d_pktseq_cw(new float[field_coder::codeword_length(pktseq_code)])
{
}

pr_deframer_gsm_vcc::~pr_deframer_gsm_vcc()
{
}

bool pr_deframer_gsm_vcc::deframe(const rxframe& inmeta, const gr_complex* in,
		rxframe& outmeta, gr_complex* out)
{
	switch(inmeta.frame_type & GSM_FRAME_TYPEMASK)
	{
	case GSM_FRAME_SYNC:
		{
			// For sync frames, just decode the packet sequence number
			field_coder seq_coder(d_pktseq_code);
			
			modulation_softdemod_bpsk_v(d_pktseq_cw.get(), in, seq_coder.codeword_length()/2);
			modulation_softdemod_bpsk_v(d_pktseq_cw.get()+seq_coder.codeword_length()/2,
					in+seq_coder.codeword_length()/2+d_sync_code.size(), 
					seq_coder.codeword_length()/2);

			if(seq_coder.decode(d_pktseq_cw.get(), &d_pktseq) == 0)
				d_syncvalid = true;
			else
				d_syncvalid = false;
		}
		return false;

	case GSM_FRAME_DATA:
		if(d_pktseq == 0xffffffff)
			return false;

		// For data frames, fill sequence numbers and copy data
		if(d_syncvalid)
		{
			outmeta.pkt_seq = d_pktseq;
			outmeta.frame_seq = (inmeta.frame_type & GSM_FRAME_SEQMASK);
			outmeta.stamp = inmeta.stamp;

			std::copy(in, in+data_size()/2, out);
			std::copy(in+data_size()/2+d_data_code.size(), in+data_size()+d_data_code.size(), out+data_size()/2);

			return true;
		}
		else
			return false;

	default:
		cerr << format("pr_deframer_gsm_vcc: invalid frame type %d") % (inmeta.frame_type & GSM_FRAME_TYPEMASK) << std::endl;
	}

	return false;
}

