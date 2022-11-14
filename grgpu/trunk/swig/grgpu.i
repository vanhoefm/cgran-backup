/* -*- c++ -*- */

%include "gnuradio.i"			// the common stuff

%{
#include "grgpu_fir_fff_cuda.h"
#include "grgpu_add_const_ff_cuda.h"

#include "grgpu_resampler_fff_cuda.h"
#include "grgpu_resampler_ccf_cuda.h"

#include "grgpu_fft_vfc_cuda.h"
#include "grgpu_fir_filter_fff_cuda.h"
#include "grgpu_d2h_cuda.h"
#include "grgpu_d2h_c_cuda.h"
#include "grgpu_h2d_cuda.h"
#include "grgpu_h2d_c_cuda.h"
%}

%include "grgpu_fir_fff_cuda.i"
%include "grgpu_add_const_ff_cuda.i"

%include "grgpu_resampler_fff_cuda.i"
%include "grgpu_resampler_ccf_cuda.i"

%include "grgpu_fft_vfc_cuda.i"
%include "grgpu_fir_filter_fff_cuda.i"
%include "grgpu_d2h_cuda.i"
%include "grgpu_d2h_c_cuda.i"
%include "grgpu_h2d_cuda.i"
%include "grgpu_h2d_c_cuda.i"

