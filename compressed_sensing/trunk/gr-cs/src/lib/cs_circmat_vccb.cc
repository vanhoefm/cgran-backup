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

#include <cs_circmat_vccb.h>
#include <gr_io_signature.h>
#include <stdexcept>


cs_circmat_vccb_sptr
cs_make_circmat_vccb (const std::vector<char> &sequence, unsigned output_length, bool translate_zeros) {
	return cs_circmat_vccb_sptr(new cs_circmat_vccb(sequence, output_length, translate_zeros));
}


cs_circmat_vccb::cs_circmat_vccb(const std::vector<char> &sequence, unsigned output_length, bool translate_zeros) :
	gr_sync_block("circmat_vccb",
			gr_make_io_signature(1, 1, sizeof(gr_complex) * sequence.size()),
			gr_make_io_signature(1, 1, sizeof(gr_complex) * output_length)),
	d_input_length(sequence.size()), d_output_length(output_length),
	d_translate_zeros(translate_zeros),
	d_sequence(sequence)
{
	if (translate_zeros) {
		translate_zeros_to_negative();
	}
}


cs_circmat_vccb::~cs_circmat_vccb()
{
}


// FIXME refactor, increase readability and speed
int
cs_circmat_vccb::work(int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items)
{
	gr_complex *optr = (gr_complex *) output_items[0];
	gr_complex *iptr = (gr_complex *) input_items[0];

	for (int i = 0; i < noutput_items; i++) {
		for (unsigned row = 0; row < d_output_length; row++) {
			gr_complex acc(0);
			for (unsigned col = 0; col < d_input_length; col++) {
                                switch (d_sequence[(col + d_input_length - row) % d_input_length]) {
                                        case 1:
                                                acc += iptr[col];
                                                break;

                                        case -1:
                                                acc -= iptr[col];
                                                break;
                                }
			}
                        optr[row] = acc;
		}
                optr += d_output_length;
		iptr += d_input_length;
	}

	return noutput_items;
}


void
cs_circmat_vccb::translate_zeros_to_negative()
{
	for (unsigned i = 0; i < d_sequence.size(); i++) {
		if (d_sequence[i] == 0) {
			d_sequence[i] = -1;
		}
	}
}

