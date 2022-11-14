/* -*- c++ -*- */
/*
 * Copyright 2010 Communications Engineering Lab, KIT
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the Gorder General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * Gorder General Public License for more details.
 * 
 * You should have received a copy of the Gorder General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <itpp/base/bessel.h>
#include <itpp_bessel.h>
#include <gr_io_signature.h>

/************** Bessel function of first kind  ******************************/
itpp_besselj_ff_sptr 
itpp_make_besselj_ff (int order)
{
	if (order < 0) {
		throw std::invalid_argument("itpp_besselj_ff: order must be non-negative.");
	}
  return itpp_besselj_ff_sptr (new itpp_besselj_ff (order));
}


itpp_besselj_ff::itpp_besselj_ff (int order)
  : gr_sync_block ("besselj_ff",
	      gr_make_io_signature (1, 1, sizeof (float)),
	      gr_make_io_signature (1, 1, sizeof (float))),
	d_order(order)
{
}


itpp_besselj_ff::~itpp_besselj_ff ()
{
}


int 
itpp_besselj_ff::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	const float *in = (const float *) input_items[0];
	float *out = (float *) output_items[0];

	for (int i = 0; i < noutput_items; i++) {
		out[i] = (float) itpp::besselj(d_order, in[i]);
	}

	return noutput_items;
}

bool
itpp_besselj_ff::set_order(int order)
{
	if (order < 0) {
		return false;
	}
	d_order = order;

	return true;
}

/************** Modified Bessel function of first kind  ***********************/
itpp_besseli_ff_sptr 
itpp_make_besseli_ff (int order)
{
	if (order < 0) {
		throw std::invalid_argument("itpp_besseli_ff: order must be non-negative.");
	}
  return itpp_besseli_ff_sptr (new itpp_besseli_ff (order));
}


itpp_besseli_ff::itpp_besseli_ff (int order)
  : gr_sync_block ("besseli_ff",
	      gr_make_io_signature (1, 1, sizeof (float)),
	      gr_make_io_signature (1, 1, sizeof (float))),
	d_order(order)
{
}


itpp_besseli_ff::~itpp_besseli_ff ()
{
}


int 
itpp_besseli_ff::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	const float *in = (const float *) input_items[0];
	float *out = (float *) output_items[0];

	for (int i = 0; i < noutput_items; i++) {
		out[i] = (float) itpp::besseli(d_order, in[i]);
	}

	return noutput_items;
}

bool
itpp_besseli_ff::set_order(int order)
{
	if (order < 0) {
		return false;
	}
	d_order = order;

	return true;
}

