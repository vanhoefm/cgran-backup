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
#ifndef INCLUDED_MODULATION_H
#define INCLUDED_MODULATION_H

#include <gr_complex.h>
#include "modulation_type.h"

/**
 * \brief Returns the number of bits used per symbol for a modulation
 * scheme.
 *
 * \ingroup prim
 * \param modulation modulation type
 * \returns number of bits per symbol
 */
int modulation_bits_per_symbol(modulation_type modulation);

/**
 * \brief Primitive hard demodulation function for BPSK.
 *
 * \ingroup prim
 * \p len symbols from \p in are demodulated and stored in the space
 * pointed to by \p out. The used constellation mapping is [-1, 1].
 * \param out pointer to demodulated bits
 * \param in pointer to complex symbols
 * \param len number of symbols to demodulate
 */
void modulation_demod_bpsk(char* out, const gr_complex* in, int len);

/**
 * \brief Primitive hard demodulation function for QPSK.
 *
 * \ingroup prim
 * Not implemented.
 * \param out pointer to demodulated bits
 * \param in pointer to complex symbols
 * \param len number of symbols to demodulate
 */
void modulation_demod_qpsk(char* out, const gr_complex* in, int len);

/**
 * \brief Primitive soft demodulation function for BPSK.
 *
 * \ingroup prim
 * \p len symbols from \p in are demodulated and stored in the space
 * pointed to by \p out. The soft bit information is the real part of
 * the symbol.
 * \param out pointer to demodulated soft bits
 * \param in pointer to complex symbols
 * \param len number of symbols to demodulate
 */
void modulation_softdemod_bpsk_v(float* out, const gr_complex* in, int len);

/**
 * \brief Primitive soft demodulation function for QPSK.
 *
 * \ingroup prim
 * Not implemented.
 * \param out pointer to demodulated soft bits
 * \param in pointer to complex symbols
 * \param len number of symbols to demodulate
 */
void modulation_softdemod_qpsk_v(float* out, const gr_complex* in, int len);

#endif

