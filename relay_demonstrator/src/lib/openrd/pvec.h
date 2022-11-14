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
#ifndef INCLUDED_PVEC_H
#define INCLUDED_PVEC_H

#include <cstring>

/**
 * \brief Returns the smallest power-of-two that is at least as large as 
 * \p size.
 *
 * \ingroup prim
 */
int pvec_alloc_size(int size);

/**
 * \brief Pads a vector with zeroes.
 *
 * \ingroup prim
 * \param vec Pointer to data vector.
 * \param size Allocation size of vector.
 * \param len Size of the actual data.
 */
inline void pvec_pad(void* vec, int size, int len) { std::memset((char*)vec+len, 0, size-len); }

#endif

