/* -*- c++ -*- */
%include "gnuradio.i"			// the common stuff

%{
#include "specest_stream_to_vector_overlap.h"
#include "specest_moving_average_vff.h"
#include "specest_adaptiveweighting_vff.h"
#include "specest_pad_vector.h"
#include "specest_reciprocal_ff.h"
#include "specest_welch.h"
#include "specest_arburg_vcc.h"
#include "specest_arfcov_vcc.h"
#include "specest_burg.h"
#include "specest_fcov.h"
%}

%include "specest_stream_to_vector_overlap.i"
%include "specest_moving_average_vff.i"
%include "specest_adaptiveweighting_vff.i"
%include "specest_pad_vector.i"
%include "specest_reciprocal_ff.i"
%include "specest_welch.i"
%include "specest_arburg_vcc.i"
%include "specest_arfcov_vcc.i"
%include "specest_burg.i"
%include "specest_fcov.i"

