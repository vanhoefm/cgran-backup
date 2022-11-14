/* -*- c++ -*- */
/*
 * Copyright 2010 Communications Engineering Lab, KIT
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

#include <itpp_reedsolomon.h>
#include <gr_io_signature.h>

#define M_TO_N(m) ((1 << m) - 1)
#define MT_TO_K(m, t) ((1 << m) - 2 * t - 1)

bool itppi_reedsolomon_check_arguments(int m, int t)
{
	if (m < 1 || t < 1 || MT_TO_K(m, t) < 0) {
		return false;
	}
	return true;
}

/*********** Base class ****************************************************/
itpp_reedsolomon_coder::itpp_reedsolomon_coder(int m, int t, bool systematic)
	: d_n(M_TO_N(m)), d_k(MT_TO_K(m, t)), d_m(m)
{
	d_reedsolomon_coder = new itpp::Reed_Solomon(m, t, systematic); 
}

/*********** Encoder *******************************************************/
itpp_reedsolomon_encoder_vbb_sptr 
itpp_make_reedsolomon_encoder_vbb (int m, int t, bool systematic)
{
	if (!itppi_reedsolomon_check_arguments(m, t)) {
		throw std::invalid_argument("itpp_reedsolomon_encoder_vbb: Invalid values (m, t).");
	}
	return itpp_reedsolomon_encoder_vbb_sptr (new itpp_reedsolomon_encoder_vbb (m, t, systematic));
}


itpp_reedsolomon_encoder_vbb::itpp_reedsolomon_encoder_vbb (int m, int t, bool systematic)
  : gr_sync_block ("reedsolomon_encoder_vbb",
	      gr_make_io_signature (1, 1, sizeof(char) * MT_TO_K(m, t) * m),
	      gr_make_io_signature (1, 1, sizeof(char) * M_TO_N(m) * m)),
	itpp_reedsolomon_coder(m, t, systematic)
{}


int 
itpp_reedsolomon_encoder_vbb::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	itpp::bin *in = (itpp::bin *) input_items[0];
	unsigned char *out = (unsigned char *) output_items[0];
	itpp::bvec v_in(in, d_k * d_m *  noutput_items * sizeof(char));
	itpp::bvec v_out(d_n * d_m * noutput_items);

	// Run encoder
	d_reedsolomon_coder->encode(v_in, v_out);

	// Copy to output_items
	memcpy(out, v_out._data(), d_n * noutput_items * sizeof(char));

	return noutput_items;
}


/*********** Decoder *******************************************************/
itpp_reedsolomon_decoder_vbb_sptr 
itpp_make_reedsolomon_decoder_vbb (int m, int t, bool systematic)
{
	if (!itppi_reedsolomon_check_arguments(m, t)) {
		throw std::invalid_argument("itpp_reedsolomon_decoder_vbb: Invalid values (m, t).");
	}
	return itpp_reedsolomon_decoder_vbb_sptr (new itpp_reedsolomon_decoder_vbb (m, t, systematic));
}


itpp_reedsolomon_decoder_vbb::itpp_reedsolomon_decoder_vbb (int m, int t, bool systematic)
  : gr_sync_block ("reedsolomon_decoder_vbb",
	      gr_make_io_signature (1, 1, sizeof(char) * M_TO_N(m) * m),
	      gr_make_io_signature (1, 1, sizeof(char) * MT_TO_K(m, t) * m)),
	itpp_reedsolomon_coder(m, t, systematic)
{}


int 
itpp_reedsolomon_decoder_vbb::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	itpp::bin *in = (itpp::bin *) input_items[0];
	unsigned char *out = (unsigned char *) output_items[0];
	itpp::bvec v_in(in, d_n * d_m *  noutput_items);
	itpp::bvec v_out(d_k * d_m * noutput_items);

	// Run decoder
	d_reedsolomon_coder->decode(v_in, v_out);

	// Copy to output_items
	memcpy(out, v_out._data(), d_k * noutput_items * sizeof(char));

	return noutput_items;
}

