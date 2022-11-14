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

#include "pr_framer_gsm_vbb.h"

#include <gr_io_signature.h>
#include "stream_p.h"
#include <cstring>
#include <algorithm>
#include "field_coder.h"

pr_framer_gsm_vbb_sptr pr_make_framer_gsm_vbb(int frame_size, field_code_type pktseq_code,
		const std::vector<char>& sync_code, const std::vector<char>& data_code)
		
{
	return pr_framer_gsm_vbb_sptr(new pr_framer_gsm_vbb(frame_size, pktseq_code, sync_code, data_code));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_framer_gsm_vbb::pr_framer_gsm_vbb(int frame_size, field_code_type pktseq_code,
		const std::vector<char>& sync_code, const std::vector<char>& data_code) :
	pr_framer_vbb(frame_size-data_code.size(), frame_size),
	d_pktseq_code(pktseq_code), d_sync_code(sync_code), d_data_code(data_code),
	d_insertsync(true)
{
	int seqlen = field_coder::codeword_length(pktseq_code);

	if(seqlen%2 != 0)
		throw "pr_framer_gsm_vbb: length of coded sequence number must be even";

	if(seqlen+sync_code.size() != (unsigned int)frame_size)
		throw "pr_framer_gsm_vbb: synchronization frame has invalid length";

	d_pktseq_cw.reset(new char[seqlen]);
}

pr_framer_gsm_vbb::~pr_framer_gsm_vbb()
{
}

int pr_framer_gsm_vbb::general_work(int noutput_items,
		gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, ninput_items, input_items, output_items);

	const_stream_p<const txmeta,const char> in(input_items[0], data_size(), ninput_items[0]);
	stream_p<txmeta,char> out(output_items[0], frame_size(), noutput_items);

	while(!in.atend() && !out.atend())
	{
		out.meta().pkt_seq = in.meta().pkt_seq;
		out.meta().frame_seq = in.meta().frame_seq;
		out.meta().data_valid = in.meta().data_valid;

		if(in.meta().frame_seq == 0 && d_insertsync)
		{
			d_insertsync = false;

			field_coder seq_coder(d_pktseq_code);
			
			seq_coder.code(in.meta().pkt_seq, d_pktseq_cw.get());

			std::copy(d_pktseq_cw.get(), d_pktseq_cw.get()+seq_coder.codeword_length()/2, out.data());
			std::copy(d_sync_code.begin(), d_sync_code.end(), out.data()+seq_coder.codeword_length()/2);
			std::copy(d_pktseq_cw.get()+seq_coder.codeword_length()/2, 
					d_pktseq_cw.get()+seq_coder.codeword_length(), 
					out.data()+seq_coder.codeword_length()/2+d_sync_code.size());
		}
		else
		{
			d_insertsync = true;

			std::copy(in.data(), in.data()+data_size()/2, out.data());
			std::copy(d_data_code.begin(), d_data_code.end(), out.data()+data_size()/2);
			std::copy(in.data()+data_size()/2, in.data()+data_size(), out.data()+data_size()/2+d_data_code.size());

			in.next();
		}

		out.next();
	}

	consume(0, in.current());

	work_used(this, 0, in.current());
	work_exit(this, out.current());

	return out.current();
}

