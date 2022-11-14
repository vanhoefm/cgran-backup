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
#ifndef INCLUDED_RM_H
#define INCLUDED_RM_H

typedef struct rm
{
	int r;
	int m;

	int k;
	int n;

	char** G;
	int* mlvsize;
	unsigned short**** mlv;
} rm_t;

/**
 * \brief Encodes a message using the Reed-Muller code rm.
 *
 * \ingroup prim
 *
 * \param rm Reed-Muller code
 * \param m source message of rm->k bits
 * \param e encoded word of rm->n bits
 */
void rm_encode(const rm_t* rm, char* m, char* e);

/**
 * \brief Decodes a message encoded with the Reed-Muller code rm.
 *
 * \ingroup prim
 *
 * \param rm Reed-Muller code
 * \param x noisy codeword of rm->n bits
 * \param d decoded message of rm->k bits
 */
void rm_decode(const rm_t* rm, char* x, char* d);

#endif

