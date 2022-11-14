#include <itpp/comm/channel.h>

// This might change in IT++...
namespace itpp {
	enum CHANNEL_PROFILE {
		ITU_Vehicular_A, ITU_Vehicular_B, ITU_Pedestrian_A, ITU_Pedestrian_B,
		COST207_RA, COST207_RA6,
		COST207_TU, COST207_TU6alt, COST207_TU12, COST207_TU12alt,
		COST207_BU, COST207_BU6alt, COST207_BU12, COST207_BU12alt,
		COST207_HT, COST207_HT6alt, COST207_HT12, COST207_HT12alt,
		COST259_TUx, COST259_RAx, COST259_HTx
	};

	enum FADING_TYPE { Independent, Static, Correlated };

	enum CORRELATED_METHOD { Rice_MEDS, IFFT, FIR };

	enum DOPPLER_SPECTRUM {
		Jakes = 0, J = 0, Classic = 0, C = 0,
		GaussI = 1, Gauss1 = 1, GI = 1, G1 = 1,
		GaussII = 2, Gauss2 = 2, GII = 2, G2 = 2
	};
}


GR_SWIG_BLOCK_MAGIC(itpp,channel_tdl_vcc);

itpp_channel_tdl_vcc_sptr itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile);
itpp_channel_tdl_vcc_sptr itpp_make_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time);

class itpp_channel_tdl_vcc : public gr_sync_block
{
 private:
	itpp_channel_tdl_vcc (unsigned vlen, unsigned vpad, std::vector<double> &avg_power_db, std::vector<int> &delay_profile);
	itpp_channel_tdl_vcc (unsigned vlen, unsigned vpad, const itpp::CHANNEL_PROFILE chan_profile, double sample_time);

 public:
	void set_channel_profile (const std::vector<float> &avg_power_dB, const std::vector<int> &delay_prof);
	void set_channel_profile_uniform (int no_taps);
	void set_channel_profile_exponential (int no_taps);
	//void set_channel_profile (const Channel_Specification &channel_spec, double sampling_time);
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
	//double get_sampling_time () const;
	//double get_sampling_rate () const;
};

