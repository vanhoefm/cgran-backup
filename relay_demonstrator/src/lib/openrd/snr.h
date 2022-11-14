/* -*- c++ -*- */
/*
 * Copyright 2011 Anton Blad.
 * 
 * This file is part of OpenRD
 * 
 * OpenRD is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * OpenRD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
#ifndef INCLUDED_SNR_H
#define INCLUDED_SNR_H

#include <gr_complex.h>

/**
 * \brief Estimates the SNR of a number of symbols using the M2M4 algorithm.
 *
 * \ingroup prim
 * The function works for BPSK and QPSK modulated data.
 * \param data pointer to vector of symbols
 * \param len number of symbols
 * \returns estimated signal-to-noise ratio (not in dB)
 */
float snr_estimate(const gr_complex* data, int len);

#endif

