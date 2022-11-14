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

#include <itpp_egolay.h>
#include <gr_io_signature.h>

static const int EGOLAY_K = 12;
static const int EGOLAY_N = 24;


/*************** Encoder ****************************************************/
itpp_egolay_encoder_vbb_sptr 
itpp_make_egolay_encoder_vbb ()
{
	return itpp_egolay_encoder_vbb_sptr (new itpp_egolay_encoder_vbb ());
}


itpp_egolay_encoder_vbb::itpp_egolay_encoder_vbb ()
  : gr_sync_block ("egolay_encoder_vbb",
	      gr_make_io_signature (1, 1, sizeof(char) * EGOLAY_K),
	      gr_make_io_signature (1, 1, sizeof(char) * EGOLAY_N))
{}


int 
itpp_egolay_encoder_vbb::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	const itpp::bin *in = (itpp::bin *) input_items[0];
	unsigned char *out = (unsigned char *) output_items[0];
	itpp::bvec v_in(in, EGOLAY_K * noutput_items * sizeof(char));
	itpp::bvec v_out(EGOLAY_N * noutput_items * sizeof(char));

	// Run encoder
	d_egolay_coder->encode(v_in, v_out);

	// Copy to output_items
	memcpy(out, v_out._data(), EGOLAY_N * noutput_items * sizeof(char));

	return noutput_items;
}


/*************** Decoder ****************************************************/
itpp_egolay_decoder_vbb_sptr 
itpp_make_egolay_decoder_vbb ()
{
	return itpp_egolay_decoder_vbb_sptr (new itpp_egolay_decoder_vbb ());
}


itpp_egolay_decoder_vbb::itpp_egolay_decoder_vbb ()
  : gr_sync_block ("egolay_decoder_vbb",
	      gr_make_io_signature (1, 1, sizeof(char) * EGOLAY_N),
	      gr_make_io_signature (1, 1, sizeof(char) * EGOLAY_K))
{}


int 
itpp_egolay_decoder_vbb::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	const itpp::bin *in = (itpp::bin *) input_items[0];
	unsigned char *out = (unsigned char *) output_items[0];
	itpp::bvec v_in(in, EGOLAY_N * noutput_items * sizeof(char));
	itpp::bvec v_out(EGOLAY_K * noutput_items * sizeof(char));

	// Run decoder
	d_egolay_coder->decode(v_in, v_out);

	// Copy to output_items
	memcpy(out, v_out._data(), EGOLAY_K * noutput_items * sizeof(char));

	return noutput_items;
}

