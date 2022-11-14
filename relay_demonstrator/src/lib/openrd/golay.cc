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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "golay.h"

#define GOLAY_DEBUG

extern int golay_parity[];
extern int golay_numones[];

int golay_encode(int m)
{
	m &= 4095;
	return (m << 12) | golay_parity[m];
}

int golay_decode(int x)
{
	int m1 = (x >> 12) & 4095;
	int x1 = (x & 4095);
	int t;

	if(golay_numones[golay_parity[m1] ^ x1] <= 3)
		return m1;

	if(golay_numones[m1 ^ golay_parity[x1]] <= 3)
		return golay_parity[x1];

	t = 1;
	for(int i = 0; i < 12; i++)
	{
		if(golay_numones[golay_parity[m1^t] ^ x1] <= 3)
			return m1^t;
		t <<= 1;
	}

	t = 1;
	for(int i = 0; i < 12; i++)
	{
		if(golay_numones[m1 ^ golay_parity[x1^t]] <= 3)
			return golay_parity[x1^t];
		t <<= 1;
	}

	return -1;
}

int golay_numchanged(int m, int x)
{
	int n = 0;
	int t = golay_encode(m)^x;

	n += golay_numones[(t >> 12) & 4095];
	n += golay_numones[t & 4095];

	return n;
}

