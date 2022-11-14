%%% File    : dttsp_rec.hrl
%%% Author  : Frank Brickle <brickle@lambda>
%%% Description : 
%%% Created : 26 Jan 2009 by Frank Brickle <brickle@lambda>

%%% Record definitions for individual sdr-core params.

-record(run_rec,{val}). % run_mode
-record(trx_rec,{val}). % trx_mode
-record(ring_buffer_offset_rec,{val}). % int
-record(test_rec,
	{flag,
	 mode, % test_mode
	 tone_amp,
	 tone_freq,
	 twotone_a_amp,
	 twotone_a_freq,
	 twotone_b_amp,
	 twotone_b_freq,
	 noise_amp]).
-record(spec_rec,
	{rxk,
	 polyphase,
	 window,
	 type, % spec_mode
	 scale}).
-record(rx_listen_rec,{val}).
-record(rx_count_rec,{val}).
-record(rx_act_rec,{val}).
-record(bin_rec,{val}).
-record(lms_rec,
	{flag,
	 adaptive_filter_size,
	 delay,
	 adaptation_rate,
	 leakage}).
-record(blk_lms_rec,{flag,adaptation_rate}).
-record(compand_rec,{flag,factor}).
-record(dc_block_rec,{val}).
-record(filter_rec,{lo,hi}).
-record(gain_rec,{in,out}).
-record(geq_rec,{flag,size,preamp,gains}).
-record(sdr_mode_rec,{val}).
-record(nb_rec,{flag,thresh}).
-record(buflen_rec,{val}).
-record(osc_rec,{freq}).
-record(agc_rec,
	{flag,
	 gain_bottom,
	 gain_fix,
	 gain_limit,
	 gain_top,
	 fastgain_bottom,
	 fastgain_fix,
	 fastgain_limit,
	 attack,
	 decay,
	 fastattack,
	 fastdecay,
	 fasthangtime,
	 hangthresh,
	 hangtime,
	 slope}).
-record(pan_rec,{val}).
-record(sdrom_rec,{flag,thresh}).
-record(spot_tone_rec,{flag,gain,freq,rise,fall}).
-record(squelch_rec,{flag,thresh}).
-record(compress_rec,{flag,k,maxgain}).
-record(carrier_level_rec,{val}).
-record(waveshape_rec,{flag,other}).
-record(iq_eq_rec,{phase,gain}).
