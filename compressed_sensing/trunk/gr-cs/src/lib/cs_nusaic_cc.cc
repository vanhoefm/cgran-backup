/* -*- c++ -*- */
/*
 * Copyright 2009 Institut fuer Nachrichtentechnik / Uni Karlsruhe
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cs_nusaic_cc.h>
#include <gr_io_signature.h>
#include <stdexcept>

cs_nusaic_cc_sptr
cs_make_nusaic_cc (unsigned compression) {
	return cs_nusaic_cc_sptr(new cs_nusaic_cc(compression));
}


cs_nusaic_cc::cs_nusaic_cc (unsigned compression) :
	gr_block("nusaic_cc",
			  gr_make_io_signature2(2, 2, sizeof(gr_complex), sizeof(int)),
		          gr_make_io_signature(1, 1, sizeof(gr_complex))),
	d_compression(compression)
{
}


cs_nusaic_cc::~cs_nusaic_cc()
{
}


void
cs_nusaic_cc::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
        ninput_items_required[0] = noutput_items * d_compression;
        ninput_items_required[1] = noutput_items;
}


int
cs_nusaic_cc::general_work(int noutput_items,
            gr_vector_int &ninput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items)
{
	gr_complex *samples = (gr_complex *) input_items[0];
	int *offsets = (int *) input_items[1];

	gr_complex *out = (gr_complex *) output_items[0];

        if (ninput_items[0] < noutput_items * (int) d_compression) {
                noutput_items = (int) ninput_items[0] / d_compression;
        }
        if (ninput_items[1] < noutput_items) {
                noutput_items = ninput_items[1];
        }

	for (int i = 0; i < noutput_items; i++) {
		if (offsets[i] < 0 || offsets[i] >= (int) d_compression) {
			throw std::runtime_error("Sample offset was out of range (must be in [0;compression-1]).");
		}
		out[i] = samples[i * d_compression + offsets[i]];
	}

        consume(0, noutput_items * d_compression);
        consume(1, noutput_items);

	return noutput_items;
}

