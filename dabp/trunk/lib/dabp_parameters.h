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

#ifndef INCLUDED_DABP_PARAMETERS_H
#define INCLUDED_DABP_PARAMETERS_H

#include <gr_block.h>

class dabp_parameters;
typedef boost::shared_ptr<dabp_parameters> dabp_parameters_sptr;
dabp_parameters_sptr dabp_make_parameters(int mode);

class dabp_parameters
{
    friend dabp_parameters_sptr dabp_make_parameters(int mode);
    
private:
    // OFDM fundamental parameters
    static const float elementary_freq = 2048000; // f=1/T, T: the elementary period defined at p 145
    static const float sampling_freq   = 2000000; // fa, sampling rate of usrp
private:
    static const int num_symbols_per_frame[4]; // L
    static const int num_carriers[4]; // K
    static const int inv_carrier_spacing[4]; // Tu/T
    static const int guard_interval[4]; // delta/T
    static const int null_symbol_duration[4]; // Tnull/T
    static const int fic_symbols_per_frame[4]; // number of ofdm symbols for FIC in each frame
    
    static const int cifsz=55296; // size of CIF (bits @ output of time interleaver), independent of mode
    
private:
    int d_mode;
    int d_L, d_K, d_Tu, d_delta, d_Tnull, d_Tf, d_Ts; // these are in the unit of T
    int d_Nfft, d_Ndel, d_Nnull, d_Nfrm; // these are in the unit of 1/fa, the sampling interval
    int d_Nficsyms, d_Nmscsyms;
public:
    float get_f() const { return elementary_freq; }
    float get_fa() const { return sampling_freq; }
    int get_mode() const { return d_mode; }
    int get_L() const { return d_L; }
    int get_K() const { return d_K; }
    int get_Tu() const { return d_Tu; }
    int get_delta() const { return d_delta; }
    int get_Tnull() const { return d_Tnull; }
    int get_Tf() const { return d_Tf; }
    int get_Ts() const { return d_Ts; }
    int get_Nnull() const { return d_Nnull; }
    int get_Nfft() const { return d_Nfft; }
    int get_Ndel() const { return d_Ndel; }
    int get_Nfrm() const { return d_Nfrm; }
    int get_Nficsyms() const { return d_Nficsyms; }
    int get_Nmscsyms() const { return d_Nmscsyms; }
    int get_cifsz() const { return cifsz; }
public:
    dabp_parameters(int mode);
public:
    ~dabp_parameters();
    
};

#endif //INCLUDED_DABP_PARAMETERS_H
