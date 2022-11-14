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
#ifndef INCLUDED_RM_2_6_H
#define INCLUDED_RM_2_6_H

/**
 * \brief Encodes a message using the (2,6) Reed-Muller code.
 *
 * \ingroup prim
 *
 * \param m source message of 22 bits
 * \param e encoded word of 64 bits
 */
void rm_2_6_encode(char* m, char* e);

/**
 * \brief Decodes a message encoded with the (2,6) Reed-Muller code.
 *
 * \ingroup prim
 * This function corrects all errors of at most seven bits in the vector.
 *
 * \param x noisy codeword
 * \param d decoded message
 */
void rm_2_6_decode(char* x, char* d);

#endif

