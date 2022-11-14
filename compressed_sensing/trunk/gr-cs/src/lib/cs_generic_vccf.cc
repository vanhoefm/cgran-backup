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

#include <cs_generic_vccf.h>
#include <gr_io_signature.h>
#include <stdexcept>


cs_generic_vccf_sptr
cs_make_generic_vccf (const std::vector<std::vector<float> > &comp_matrix) {
	return cs_generic_vccf_sptr(new cs_generic_vccf(comp_matrix));
}

cs_generic_vccf::cs_generic_vccf (const std::vector<std::vector<float> > &comp_matrix) :
	gr_sync_block("generic_vccf",
			gr_make_io_signature(1, 1, sizeof(gr_complex) * comp_matrix[0].size()),
			gr_make_io_signature(1, 1, sizeof(gr_complex) * comp_matrix.size())),
	d_input_length(comp_matrix[0].size()), d_output_length(comp_matrix.size()),
	d_comp_matrix(comp_matrix.size())
{
	for (unsigned i = 0; i < d_output_length; i++) {
		d_comp_matrix[i].assign(comp_matrix[i].begin(), comp_matrix[i].end());
	}
}


cs_generic_vccf::~cs_generic_vccf()
{
}


int
cs_generic_vccf::work(int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items)
{
	gr_complex *optr = (gr_complex *) output_items[0];
	gr_complex *iptr = (gr_complex *) input_items[0];

	for (int i = 0; i < noutput_items; i++) {
		for (unsigned row = 0; row < d_output_length; row++) {
			gr_complex acc(0);
			for (unsigned col = 0; col < d_input_length; col++) {
				acc += iptr[col] * d_comp_matrix[row][col];
			}
                        optr[row] = acc;
		}
                optr += d_output_length;
		iptr += d_input_length;
	}

	return noutput_items;
}

