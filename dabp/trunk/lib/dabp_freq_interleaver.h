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
#ifndef INCLUDED_DABP_FREQ_INTERLEAVER_H
#define INCLUDED_DABP_FREQ_INTERLEAVER_H

/*!
 * \brief DAB frequency interleaver EN 300 401 Section 14.6
 * \ingroup dabp
 * The QPSK symbols shall be re-ordered according to the following relation:
 * y_k=q_n, where k=F(n) and F is defined differently for the four modes
 * k is frequency index [-K/2, K/2]\0, n is the index of the QPSK symbol [0,K-1]
 * Note that frequency interleaving is only performed within each OFDM symbol
 */
class dabp_freq_interleaver
{
    private:
    int *d_F; // the F table. the index to F is n, the value in F is k
    int *d_A; // aux A table
    int d_N, d_K; // ideal fft length and used carriers
	int d_Nfft; // actual fft length used in the receiver
    public:
    dabp_freq_interleaver(int mode=1, int Nfft=2000);
    ~dabp_freq_interleaver();
    int interleave(int idx); // input n output k modulo Nfft
};
#endif // INCLUDED_DABP_FREQ_INTERLEAVER_H

