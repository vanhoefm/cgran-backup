/* -*- c++ -*- */
/*
 * Copyright 2004,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include <iostream>
#include "dabp_parameters.h"

dabp_parameters_sptr
dabp_make_parameters (int mode)
{
  return dabp_parameters_sptr (new dabp_parameters (mode));
}

// parameters below are from Table 38 on page 145 of EN 300 401
const int dabp_parameters::num_symbols_per_frame[4]={76,76,153,76}; // L
const int dabp_parameters::num_carriers[4]={1536,384,192,768}; // K
const int dabp_parameters::null_symbol_duration[4]={2656,664,345,1328}; // Tnull/T
const int dabp_parameters::inv_carrier_spacing[4]={2048,512,256,1024}; // Tu/T
const int dabp_parameters::guard_interval[4]={504,126,63,252}; // delta/T
const int dabp_parameters::fic_symbols_per_frame[4]={3,3,8,3};

// Note that Ts=Tu+delta, Tf=Tnull+Ts*L. So they are not independant parameters

dabp_parameters::dabp_parameters(int mode) : d_mode(mode)
{
    assert(mode>=1 && mode<=4); // validity check
    // parameters from Table 38
    d_L=num_symbols_per_frame[mode-1];
    d_K=num_carriers[mode-1];
    d_Tu=inv_carrier_spacing[mode-1];
    d_delta=guard_interval[mode-1];
    d_Tnull=null_symbol_duration[mode-1];
    d_Ts = d_Tu + d_delta;
    d_Tf = d_Tnull + d_Ts*d_L;
    // parameters for usrp implementation
    d_Nfft = (int)(d_Tu/elementary_freq*sampling_freq +0.5);
    d_Ndel = (int)(d_delta/elementary_freq*sampling_freq +0.5);
    d_Nnull = (int)(d_Tnull/elementary_freq*sampling_freq +0.5);
    d_Nfrm = (int)(d_Tf/elementary_freq*sampling_freq +0.5);
    
    d_Nficsyms = fic_symbols_per_frame[mode-1];
    d_Nmscsyms = d_L - 1 - d_Nficsyms;
}

dabp_parameters::~dabp_parameters()
{
}

