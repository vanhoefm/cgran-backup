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

#include "pr_constellation_decoder_cb.h"

#include <gr_io_signature.h>
#include <gr_complex.h>
#include "pvec.h"

pr_constellation_decoder_cb_sptr pr_make_constellation_decoder_cb(modulation_type modulation)
{
	return pr_constellation_decoder_cb_sptr(new pr_constellation_decoder_cb(modulation));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_constellation_decoder_cb::pr_constellation_decoder_cb(modulation_type modulation) :
	gr_sync_block("constellation_decoder_cb",
			 gr_make_io_signature(MIN_IN, MAX_IN, sizeof(gr_complex)),
			 gr_make_io_signature(MIN_OUT, MAX_OUT, pvec_alloc_size(modulation_bits_per_symbol(modulation)))),
	d_modulation(modulation)
{
}

pr_constellation_decoder_cb::~pr_constellation_decoder_cb()
{
}

int pr_constellation_decoder_cb::work(int noutput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items)
{
	work_enter(this, noutput_items, 0, input_items, output_items);

	const gr_complex* in = (const gr_complex*) input_items[0];
	char* out = (char*) output_items[0];

	switch(d_modulation)
	{
	case MODULATION_BPSK:
		modulation_demod_bpsk(out, in, noutput_items);
		break;
	case MODULATION_QPSK:
		modulation_demod_qpsk(out, in, noutput_items);
		break;
	}

	work_used(this, 0, noutput_items);
	work_exit(this, noutput_items);
	
	return noutput_items;
}

modulation_type pr_constellation_decoder_cb::modulation() const
{
	return d_modulation;
}

int pr_constellation_decoder_cb::symbol_bits() const
{
	return modulation_bits_per_symbol(d_modulation);
}

