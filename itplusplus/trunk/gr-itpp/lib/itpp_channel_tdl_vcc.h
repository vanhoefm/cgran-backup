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
#ifndef INCLUDED_ITPP_CHANNEL_TDL_vcc_H
#define INCLUDED_ITPP_CHANNEL_TDL_vcc_H

#include <itpp/comm/channel.h>
#include <gr_sync_block.h>
#include <gruel/thread.h>

class itpp_channel_tdl_vcc;

typedef boost::shared_ptr<itpp_channel_tdl_vcc> itpp_channel_tdl_vcc_sptr;

// Possible FIXME: In IT++, default values are accepted, although I don't quite see the point
itpp_channel_tdl_vcc_sptr itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile);
itpp_channel_tdl_vcc_sptr itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time);

/**
 * \brief Tapped delay line channel model.
 *
 * This works on entire signals, capsuled in vectors. Vector length is given by \p vlen.
 * Next, \p vpad zeros are added to catch the channel's delay line.
 */
class itpp_channel_tdl_vcc : public gr_sync_block
{
 private:
	friend itpp_channel_tdl_vcc_sptr itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile);
	friend itpp_channel_tdl_vcc_sptr itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time);

	itpp_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile);
	itpp_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time);

	unsigned d_vlen;
	unsigned d_vpad;
	gruel::mutex d_mutex;
	itpp::TDL_Channel *d_channel;

 public:
	~itpp_channel_tdl_vcc ();

	void set_channel_profile (const std::vector<float> &avg_power_dB, const std::vector<int> &delay_prof);
	void set_channel_profile_uniform (int no_taps);
	void set_channel_profile_exponential (int no_taps);
	//void set_channel_profile (const itpp::Channel_Specification &channel_spec, double sampling_time);
	//void set_correlated_method (itpp::CORRELATED_METHOD method);
	//void set_fading_type (itpp::FADING_TYPE fading_type);
	//void set_norm_doppler (double norm_doppler);
	//void set_LOS (const vec &relative_power, const std::vector<double> &relative_doppler = "");
	//void set_LOS_power (const std::vector<double> &relative_power);
	//void set_LOS_doppler (const std::vector<double> &relative_doppler);
	//void set_doppler_spectrum (const DOPPLER_SPECTRUM *tap_spectrum);
	//void set_doppler_spectrum (int tap_number, DOPPLER_SPECTRUM tap_spectrum);
	//void set_no_frequencies (int no_freq);
	//void set_time_offset (int offset);
	//void shift_time_offset (int no_samples);
	//void set_filter_length (int filter_length);
	//int taps () const;
	//void get_channel_profile (vec &avg_power_dB, ivec &delay_prof) const;
	//vec get_avg_power_dB () const;
	//ivec get_delay_prof () const;
	//itpp::CORRELATED_METHOD get_correlated_method () const;
	//itpp::FADING_TYPE get_fading_type () const;
	//double get_norm_doppler () const;
	//vec get_LOS_power () const;
	//vec get_LOS_doppler () const;
	//double get_LOS_power (int tap_number) const;
	//double get_LOS_doppler (int tap_number) const;
	//int get_no_frequencies () const;
	//double get_time_offset () const;
	//double calc_mean_excess_delay () const;
	//double calc_rms_delay_spread () const;
	//double get_sampling_time () const
	//double get_sampling_rate () const

	int work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
};

#endif /* INCLUDED_ITPP_CHANNEL_TDL_vcc_H */
