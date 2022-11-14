/* -*- c++ -*- */

%feature("autodoc", "1");		// generate python docstrings

%include "exception.i"
%import "gnuradio.i"			// the common stuff

%{
#include "gnuradio_swig_bug_workaround.h"	// mandatory bug fix
#include "bbn_tap.h"
#include "bbn_dpsk_demod_cb.h"
#include "bbn_crc16.h"
#include "bbn_slicer_cc.h"
#include "bbn_plcp80211_bb.h"
#include "bbn_scrambler_bb.h"
#include "bbn_firdes_barker.h"
#include <stdexcept>
%}

%include "bbn_tap.i"
%include "bbn_dpsk_demod_cb.i"
%include "bbn_crc16.i"
%include "bbn_slicer_cc.i"
%include "bbn_plcp80211_bb.i"
%include "bbn_scrambler_bb.i"
%include "bbn_firdes_barker.i"
