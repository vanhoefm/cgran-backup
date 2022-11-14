%%%-------------------------------------------------------------------
%%% File    : dttsp_types.erl
%%% Author  : Frank Brickle <brickle@lambda>
%%% Description : 
%%%
%%% Created : 26 Jan 2009 by Frank Brickle <brickle@lambda>
%%%-------------------------------------------------------------------
-module(dttsp_types).
-compile(export_all).

tlist(categories) ->
    {ok,[basic,
	 trx_mode,
	 run_mode,
	 test_mode,
	 spec_scale,
	 spec_type,
	 window,
	 sdr_mode,
	 tx_meter_type]};
tlist(basic) -> {ok,[off,on]};
tlist(trx_mode) -> {ok,[rx,tx]};
tlist(run_mode) ->
    {ok,[run_mute,run_pass,run_play,run_swch,run_test]};
tlist(test_mode) -> {ok,[tone,twotone,chirp,noise]};
tlist(spec_scale) ->
    {ok,[spec_mag,spec_pwr]};
tlist(spec_type) ->
    {ok,[spec_semi_raw,
	 spec_pre_filt,
	 spec_post_filt,
	 spec_post_agc,
	 spec_post_det,
	 spec_premod]};
tlist(window) ->
    {ok,[rectangular,
	 hanning,
	 welch,
	 parzen,
	 bartlett,
	 hamming,
	 blackman2,
	 blackman3,
	 blackman4,
	 exponential,
	 riemann]};
tlist(sdr_mode) ->
    {ok,[lsb,usb,dsb,cwl,cwu,fmn,am,digu,spec,digl,sam,drm]};
tlist(tx_meter_type) ->
    {ok, [tx_pwr,
	  tx_leveler,
	  tx_comp,
	  tx_cpdr,
	  tx_wavs,
	  tx_lvl_g,
	  tx_none]};

tlist(all) ->
    {ok,Cat} = tlist(categories),
    All = lists:map(fun(C) -> {ok,L} = tlist(C),{C,L} end, Cat),
    {ok,All};
tlist(_) -> error.

t(off) -> 0;
t(on)  -> 1;
t({basic,0}) -> off;
t({basic,1}) -> on;

t(rx) -> 0;
t(tx) -> 1;
t({trx_mode,0}) -> rx;
t({trx_mode,1}) -> tx;

t(run_mute) -> 0;
t(run_pass) -> 1;
t(run_play) -> 2;
t(run_swch) -> 3;
t(run_test) -> 4;
t({run_mode,0}) -> run_mute;
t({run_mode,1}) -> run_pass;
t({run_mode,2}) -> run_play;
t({run_mode,3}) -> run_swch;
t({run_mode,4}) -> run_test;

t(tone)-> 0;
t(twotone)-> 1;
t(chirp)-> 2;
t(noise)-> 3;
t({test_mode,0})-> tone;
t({test_mode,1})-> twotone;
t({test_mode,2})-> chirp;
t({test_mode,3})-> noise;

t(spec_mag) -> 0;
t(spec_pwr) -> 1;
t({spec_scale,0}) -> spec_mag;
t({spec_scale,1}) -> spec_pwr;

t(spec_semi_raw)  -> 0;
t(spec_pre_filt)  -> 1;
t(spec_post_filt) -> 2;
t(spec_post_agc)  -> 3;
t(spec_post_det)  -> 4;
t(spec_premod)    -> 4;
t({spec_type,0}) -> spec_semi_raw;
t({spec_type,1}) -> spec_pre_filt;
t({spec_type,2}) -> spec_postfilt;
t({spec_type,3}) -> spec_post_agc;
t({spec_type,4}) -> spec_post_det;

t(rectangular) -> 1;
t(hanning)     -> 2;
t(welch)       -> 3;
t(parzen)      -> 4;
t(bartlett)    -> 5;
t(hamming)     -> 6;
t(blackman2)   -> 7;
t(blackman3)   -> 8;
t(blackman4)   -> 9;
t(exponential) -> 10;
t(riemann)     -> 11;
t({window,1})  -> rectangular;
t({window,2})  -> hanning;
t({window,3})  -> welch;
t({window,4})  -> parzen;
t({window,5})  -> bartlett;
t({window,6})  -> hamming;
t({window,7})  -> blackman2;
t({window,8})  -> blackman3;
t({window,9})  -> blackman4;
t({window,10}) -> exponential;
t({window,11}) -> riemann;

t(tx_mic)     -> 0;
t(tx_pwr)     -> 1; 
t(tx_leveler) -> 2;
t(tx_comp)    -> 3;
t(tx_cpdr)    -> 4;
t(tx_wavs)    -> 5;
t(tx_lvl_g)   -> 6;
t(tx_none)    -> 7;
t({tx_meter_type,0}) -> tx_pwr;
t({tx_meter_type,1}) -> tx_eqtap;
t({tx_meter_type,2}) -> tx_leveler;
t({tx_meter_type,3}) -> tx_comp;
t({tx_meter_type,4}) -> tx_cpdr;
t({tx_meter_type,5}) -> tx_wavs;
t({tx_meter_type,6}) -> tx_lvl_g;
t({tx_meter_type,7}) -> tx_none;

t(lsb)  -> 0;
t(usb)  -> 1;
t(dsb)  -> 2;
t(cwl)  -> 3;
t(cwu)  -> 4;
t(fmn)  -> 5;
t(am)   -> 6;
t(digu) -> 7;
t(spec) -> 8;
t(digl) -> 9;
t(sam)  -> 10;
t(drm)  -> 11;
t({sdr_mode,0})  -> lsb;
t({sdr_mode,1})  -> usb;
t({sdr_mode,2})  -> dsb;
t({sdr_mode,3})  -> cwl;
t({sdr_mode,4})  -> cwu;
t({sdr_mode,5})  -> fmn;
t({sdr_mode,6})  -> am;
t({sdr_mode,7})  -> digu;
t({sdr_mode,8})  -> spec;
t({sdr_mode,9})  -> digl;
t({sdr_mode,10}) -> sam;
t({sdr_mode,11}) -> drm;

t(_) -> error.
