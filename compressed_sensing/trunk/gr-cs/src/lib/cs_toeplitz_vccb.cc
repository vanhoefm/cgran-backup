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

#include <cs_toeplitz_vccb.h>
#include <gr_io_signature.h>
#include <stdexcept>


cs_toeplitz_vccb_sptr
cs_make_toeplitz_vccb (unsigned input_length, unsigned output_length, const std::vector<char> &seq_history) {
	return cs_toeplitz_vccb_sptr(new cs_toeplitz_vccb(input_length, output_length, seq_history));
}
//cs_toeplitz_vccb_sptr
//cs_make_toeplitz_vccb (unsigned input_length, unsigned output_length) {
	//std::vector <char> seq_history;
	//return cs_toeplitz_vccb_sptr(new cs_toeplitz_vccb(input_length, output_length, seq_history));
//}


cs_toeplitz_vccb::cs_toeplitz_vccb (unsigned input_length, unsigned output_length, const std::vector<char> &seq_history) :
	gr_sync_block("toeplitz_vccb",
			gr_make_io_signature2(2, 2, sizeof(gr_complex) * input_length, sizeof(char) * input_length),
			gr_make_io_signature(1, 1, sizeof(gr_complex) * output_length)),
	d_input_length(input_length), d_output_length(output_length),
	d_seq_buf(output_length-1, 0)
{
	if (seq_history.size() != 0) {
		if (seq_history.size() != output_length-1) {
			throw std::invalid_argument("When giving a sequence history, it has to have exactly output_length-1 entries.");
		}

		for (unsigned i = 0; i < output_length-1; i++) {
			d_seq_buf[output_length-2-i] = seq_history[i];
		}
	}
}


cs_toeplitz_vccb::~cs_toeplitz_vccb()
{
}


int
cs_toeplitz_vccb::work(int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items)
{
	gr_complex *out = (gr_complex *) output_items[0];
	gr_complex *in_data = (gr_complex *) input_items[0];
	char *in_seq = (char *) input_items[1];

	for (int i = 0; i < noutput_items; i++) {
		for (unsigned row = 0; row < d_output_length; row++) {
			gr_complex acc(0);
			for (unsigned col = 0; col < d_input_length; col++) {
				if (comp_matrix_element(in_seq, row, col) == 1) {
					acc += in_data[col];
				} else {
					acc -= in_data[col];
				}
			}
                        out[row] = acc;
		}
		for (unsigned j = 0; j < d_output_length-1; j++) {
			d_seq_buf[j] = in_seq[d_input_length - 1 - j];
		}
                out += d_output_length;
		in_data += d_input_length;
		in_seq += d_input_length;
	}

	return noutput_items;
}


inline char
cs_toeplitz_vccb::comp_matrix_element(char *sequence, unsigned row, unsigned col)
{
	if (col >= row) {
		return sequence[col - row];
	} else {
		return d_seq_buf[row - col - 1];
	}
}

