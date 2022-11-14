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

#include <itpp_hamming.h>
#include <gr_io_signature.h>

#define M_TO_N(m) ((1 << m) - 1)
#define M_TO_K(m) ((1 << m) - m - 1)

/************** Encoder ****************************************************/
itpp_hamming_encoder_vbb_sptr 
itpp_make_hamming_encoder_vbb (short m)
{
	if (m < 1) {
		throw std::invalid_argument("itpp_hamming_encoder_vbb: m must be positive integer.");
	}
  return itpp_hamming_encoder_vbb_sptr (new itpp_hamming_encoder_vbb (m));
}


itpp_hamming_encoder_vbb::itpp_hamming_encoder_vbb (short m)
  : gr_sync_block ("hamming_encoder_vbb",
	      gr_make_io_signature (1, 1, sizeof (char) * M_TO_K(m)),
	      gr_make_io_signature (1, 1, sizeof (char) * M_TO_N(m))),
	itpp_hamming_coder(m)
{}

int 
itpp_hamming_encoder_vbb::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	const unsigned char *in = (const unsigned char *) input_items[0];
	unsigned char *out = (unsigned char *) output_items[0];
	itpp::bvec v_in(get_k() * noutput_items);
	itpp::bvec v_out(get_n() * noutput_items);

	// Copy to IT++ buffer
	memcpy(v_in._data(), in, get_k() * noutput_items * sizeof(char));

	// Run encoder
	d_hamming_coder->encode(v_in, v_out);

	// Copy to output_items
	memcpy(out, v_out._data(), get_n() * noutput_items * sizeof(char));

	return noutput_items;
}

/************** Decoder ****************************************************/
itpp_hamming_decoder_vbb_sptr 
itpp_make_hamming_decoder_vbb (short m)
{
	if (m < 1) {
		throw std::invalid_argument("itpp_hamming_decoder_vbb: m must be positive integer.");
	}
  return itpp_hamming_decoder_vbb_sptr (new itpp_hamming_decoder_vbb (m));
}


itpp_hamming_decoder_vbb::itpp_hamming_decoder_vbb (short m)
  : gr_sync_block ("hamming_decoder_vbb",
	      gr_make_io_signature (1, 1, sizeof (char) * M_TO_N(m)),
	      gr_make_io_signature (1, 1, sizeof (char) * M_TO_K(m))),
	itpp_hamming_coder(m)
{}


int 
itpp_hamming_decoder_vbb::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	itpp::bin *in = (itpp::bin *) input_items[0];
	unsigned char *out = (unsigned char *) output_items[0];
	itpp::bvec v_in(in, get_n() * noutput_items);
	itpp::bvec v_out(get_k() * noutput_items);

	// Run decoder
	d_hamming_coder->decode(v_in, v_out);

	// Copy to output_items
	memcpy(out, v_out._data(), get_k() * noutput_items * sizeof(char));

	return noutput_items;
}

