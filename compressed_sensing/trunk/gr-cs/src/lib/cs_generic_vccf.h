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

#ifndef INCLUDED_CS_GENERIC_H
#define INCLUDED_CS_GENERIC_H

#include <gr_sync_block.h>

#include <vector>

class cs_generic_vccf;
typedef boost::shared_ptr<cs_generic_vccf> cs_generic_vccf_sptr;

cs_generic_vccf_sptr
cs_make_generic_vccf (const std::vector<std::vector<float> > &comp_matrix);

/*!
 * \brief Generic, block-wise compression with custom matrix.
 *
 * Perform any kind of compressed sensing with time-discrete input signals. Note this is very ressource-
 * consuming. Input- and output-signals are vectors as not to screw up scheduling.
 *
 * The operation is equivalent to an algebraic multiplication of input vectors with the compression matrix.
 *
 * \ingroup compressedsensing
 */
class cs_generic_vccf : public gr_sync_block
{
	friend cs_generic_vccf_sptr cs_make_generic_vccf (const std::vector<std::vector<float> > &comp_matrix);

 private:
        unsigned d_input_length, d_output_length;

	std::vector<std::vector<float> > d_comp_matrix;

	cs_generic_vccf(const std::vector<std::vector<float> > &comp_matrix);

 public:
	~cs_generic_vccf();

	int work(int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);


	float get_compression() { return ((float) d_output_length) / d_input_length; };
};
#endif /* INCLUDED_CS_GENERIC_H */

