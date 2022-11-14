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

#include <itpp_channel_tdl_vcc.h>
#include <gr_io_signature.h>


itpp_channel_tdl_vcc_sptr
itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile)
{
	return itpp_channel_tdl_vcc_sptr(new itpp_channel_tdl_vcc (vlen, vpad, avg_power_db, delay_profile));
}


itpp_channel_tdl_vcc_sptr
itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time)
{
	return itpp_channel_tdl_vcc_sptr(new itpp_channel_tdl_vcc(vlen, vpad, chan_profile, sample_time));
}


itpp_channel_tdl_vcc::itpp_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile)
  : gr_sync_block ("channel_tdl_vcc",
	      gr_make_io_signature (1, 1, sizeof (gr_complex) * vlen),
	      gr_make_io_signature (1, 1, sizeof (gr_complex) * (vlen + vpad))),
	d_vlen(vlen), d_vpad(vpad)
{
	// FIXME: input is checked in TDL_Channel constructor, but see how this works w/ SWIG
	itpp::vec vec_avg_power_db((double *) &avg_power_db, avg_power_db.size());
	itpp::ivec vec_delay_profile((int *) &delay_profile, delay_profile.size());

	d_channel = new itpp::TDL_Channel(vec_avg_power_db, vec_delay_profile);
}


itpp_channel_tdl_vcc::itpp_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time)
  : gr_sync_block ("channel_tdl_vcc",
	      gr_make_io_signature (1, 1, sizeof (gr_complex) * vlen),
	      gr_make_io_signature (1, 1, sizeof (gr_complex) * (vlen + vpad))),
	d_vlen(vlen), d_vpad(vpad)
{
	// FIXME: input is checked in TDL_Channel constructor, but see how this works w/ SWIG
	itpp::Channel_Specification chanspec(chan_profile);

	d_channel = new itpp::TDL_Channel(chan_profile, sample_time);
}


itpp_channel_tdl_vcc::~itpp_channel_tdl_vcc ()
{
	delete d_channel;
}


int 
itpp_channel_tdl_vcc::work (int noutput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	const gr_complex *in = (const gr_complex *) input_items[0];
	gr_complex *out = (gr_complex *) output_items[0];
	itpp::cvec v_in(d_vlen + d_vpad);
	itpp::cvec v_out(d_vlen + d_vpad);
	v_in.zeros();

	for (int i = 0; i < noutput_items; i++) {
		// Copy to IT++ buffer
		for (int j = 0; j < d_vlen; j++) {
			v_in[i] = (std::complex<double>) in[(i*noutput_items)+j];
		}

		{ // Channel
			gruel::scoped_lock guard(d_mutex);
			d_channel->filter(v_in, v_out);
		}

		// Copy to output_items
		for (int j = 0; j < (d_vlen + d_vpad); j++) {
			out[(i*noutput_items)+j] = (gr_complex) v_out[j];
		}
	}

	return noutput_items;
}


void
itpp_channel_tdl_vcc::set_channel_profile (const std::vector<float> &avg_power_dB, const std::vector<int> &delay_prof)
{
	gruel::scoped_lock guard(d_mutex);

	// FIXME: input is checked in set_channel_profile(), but see how this works w/ SWIG
	itpp::vec vec_avg_power_db((double *) &avg_power_dB, avg_power_dB.size());
	itpp::ivec vec_delay_profile((int *) &delay_prof, delay_prof.size());

	d_channel->set_channel_profile(vec_avg_power_db, vec_delay_profile);
}


void
itpp_channel_tdl_vcc::set_channel_profile_uniform (int no_taps)
{
	gruel::scoped_lock guard(d_mutex);

	d_channel->set_channel_profile_uniform(no_taps);
}


void
itpp_channel_tdl_vcc::set_channel_profile_exponential (int no_taps)
{
	gruel::scoped_lock guard(d_mutex);

	d_channel->set_channel_profile_exponential(no_taps);
}

