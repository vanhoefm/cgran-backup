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
#ifndef INCLUDED_BITPACK_H
#define INCLUDED_BITPACK_H

/**
 * Unpacks a vector of bytes to a vector with one bit per byte, msb first.
 *
 * \ingroup prim
 * \param src Source vector.
 * \param dest Destination vector. At least 8* \p nbytes bytes must be
 * allocated.
 * \param nbytes Number of bytes to unpack.
 */
void bitpack_unpack(const unsigned char* src, unsigned char* dest, int nbytes);

/**
 * Packs a vector with one bit per byte to a byte vector, msb first.
 *
 * \ingroup prim
 * \param src Source vector.
 * \param dest Destination vector. At least \p nbytes bytes must be
 * allocated.
 * \param nbytes Number bytes to pack.
 */
void bitpack_pack(const unsigned char* src, unsigned char* dest, int nbytes);

#endif

