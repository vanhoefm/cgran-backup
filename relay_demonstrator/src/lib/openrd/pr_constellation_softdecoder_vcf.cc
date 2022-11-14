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

#include "pr_constellation_softdecoder_vcf.h"

#include <gr_io_signature.h>
#include <gr_complex.h>
#include "pvec.h"
#include "stream_p.h"

pr_constellation_softdecoder_vcf_sptr pr_make_constellation_softdecoder_vcf(modulation_type modulation, int size)
{
	return pr_constellation_softdecoder_vcf_sptr(new pr_constellation_softdecoder_vcf(modulation, size));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_constellation_softdecoder_vcf::pr_constellation_softdecoder_vcf(modulation_type modulation, int size) :
	gr_sync_block("constellation_softdecoder_vcf",
			 gr_make_io_signature(MIN_IN, MAX_IN, stream_p<rxframe,gr_complex>::alloc_size(size)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, stream_p<rxframe,float>::alloc_size(size*modulation_bits_per_symbol(modulation)))),
	d_modulation(modulation),
	d_size(size)
{
}

pr_constellation_softdecoder_vcf::~pr_constellation_softdecoder_vcf()
{
}

int pr_constellation_softdecoder_vcf::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const_stream_p<const rxframe,const gr_complex> in(input_items[0], d_size, noutput_items);
	stream_p<rxframe,float> out(output_items[0], d_size, noutput_items);

	while(!out.atend())
	{
		out.meta().power = in.meta().power;
		out.meta().pkt_seq = in.meta().pkt_seq;
		out.meta().frame_seq = in.meta().frame_seq;

		switch(d_modulation)
		{
		case MODULATION_BPSK:
			modulation_softdemod_bpsk_v(out.data(), in.data(), d_size);
			break;
		case MODULATION_QPSK:
			modulation_softdemod_qpsk_v(out.data(), in.data(), d_size);
			break;
		default:
			throw "pr_constellation_softdecoder_vcf: invalid modulation type";
		}
		
		in.next();
		out.next();
	}

	work_used(this, 0, out.current());
	work_exit(this, out.current());

	return out.current();
}

modulation_type pr_constellation_softdecoder_vcf::modulation() const
{
	return d_modulation;
}

int pr_constellation_softdecoder_vcf::symbol_bits() const
{
	return modulation_bits_per_symbol(d_modulation);
}

int pr_constellation_softdecoder_vcf::size() const
{
	return d_size;
}

