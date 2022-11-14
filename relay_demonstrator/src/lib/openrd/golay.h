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
#ifndef INCLUDED_GOLAY_H
#define INCLUDED_GOLAY_H

/**
 * \brief Encodes a message using the extended binary Golay code.
 *
 * \ingroup prim
 * Only the 12 LSBs are used, and the other bits are ignored. The returned
 * codeword contains the message left-shifted 12 positions, and the 12 LSBs
 * contain the parity bits. The encoding consists of a table lookup and is
 * very fast.
 *
 * \param m source message
 * \returns encoded word of 24 bits
 */
int golay_encode(int m);

/**
 * \brief Decodes a message encoded with the extended binary Golay code.
 *
 * \ingroup prim
 * This function corrects all errors of at most three bits in the vector.
 *
 * \param x noisy codeword
 * \returns message
 */
int golay_decode(int x);

/**
 * \brief Returns the number of changed bits in a noisy codeword.
 *
 * \ingroup prim
 * \p m is encoded using the extended binary Golay code, and compared to
 * the argument \p x.
 *
 * \param m source message
 * \param x noisy codeword
 * \returns the number of differing bits between the codeword of \p m and
 * \p x.
 */
int golay_numchanged(int m, int x);

#endif

