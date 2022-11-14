/* -*- c++ -*- */
/* 
 * Copyright 2011 Communications Engineering Lab, KIT
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

#include <gr_io_signature.h>
#include <chancoding_rmg_encoder_vii.h>
#include <chancodingi_rm_coding.h>
#include <iostream>


chancoding_rmg_encoder_vii_sptr
chancoding_make_rmg_encoder_vii (int m, int num_bits_precoded)
{
	return chancoding_rmg_encoder_vii_sptr (new chancoding_rmg_encoder_vii (m, num_bits_precoded));
}


chancoding_rmg_encoder_vii::chancoding_rmg_encoder_vii (int m, int num_bits_precoded)
	: gr_sync_block ("rmg_encoder_vii",
		gr_make_io_signature (1, 1, sizeof (int) * rm_calc_num_int(1 + m + num_bits_precoded) ),
		gr_make_io_signature (1, 1, sizeof (int) * rm_calc_num_int(1<<m))),
	d_m(m),
	d_num_int(rm_calc_num_int(1<<m)),
	d_num_rows(rm_calc_rows(2,m)),
	d_num_int_uncoded(rm_calc_num_int(1 + m + num_bits_precoded)),
	d_num_bits_precoded(num_bits_precoded),
	d_num_lin_comb_order_2(rm_binom_coeff(m,2)),
	d_num_int_precode_mat(rm_calc_num_int(d_num_lin_comb_order_2)),
	d_num_rows_precode_mat(1<<num_bits_precoded),

	d_gen_mat(rm_calc_rows(2,m) * d_num_int, 0),
	d_uncoded_temp(d_num_int_uncoded, 0),
	d_precode_mat(d_num_rows_precode_mat * d_num_int_precode_mat, 0)
{
	if(m > 12)
	{
		std::cout << std::endl << "FAILURE: m must be smaller than 13";
		abort();
	}
	if((unsigned int)num_bits_precoded > rm_calc_num_bits_precoded(m))
	{
		std::cout << std::endl << "FAILURE: Bits to precode must be less or equal to: floor(log( factorial(m) , 2))";
		abort();
	}

	rm_generate_gen_mat(2, m, &d_gen_mat[0], d_num_rows, d_num_int);
	rmg_generate_precode_mat(m, &d_precode_mat[0], d_num_rows_precode_mat, d_num_int_precode_mat);
}


chancoding_rmg_encoder_vii::~chancoding_rmg_encoder_vii ()
{
}

int chancoding_rmg_encoder_vii::get_vlen_in() {
	return rm_calc_num_int(1 + d_m + d_num_bits_precoded);
}

int chancoding_rmg_encoder_vii::get_vlen_out() {
	return rm_calc_num_int(1<<d_m);
}

int chancoding_rmg_encoder_vii::get_num_bits_in() {
	return (1 + d_m + d_num_bits_precoded);
}

int chancoding_rmg_encoder_vii::get_num_bits_out() {
	return (1<<d_m);
}

int
chancoding_rmg_encoder_vii::work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items)
{
	const unsigned int *uncoded = (const unsigned int *) input_items[0];
	unsigned int *encoded = (unsigned int *) output_items[0];
	memset((void *) encoded, 0, noutput_items * d_num_int * sizeof(int));

	for(int output_items_counter = 0; output_items_counter < noutput_items; output_items_counter++)
	{
	rmg_encode(&d_gen_mat[0], d_num_rows, d_num_int, uncoded, encoded, &d_uncoded_temp[0], d_num_bits_precoded, d_num_lin_comb_order_2, d_num_int_precode_mat, &d_precode_mat[0], d_m, d_num_int_uncoded);

		uncoded += d_num_int_uncoded;
		encoded += d_num_int;
	}

	// Tell runtime system how many output items we produced.
	return noutput_items;
}

