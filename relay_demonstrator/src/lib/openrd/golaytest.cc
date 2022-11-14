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

// Compile this test program with:
// $ g++ golaytest.cc golay.cc golaytable.cc -o golaytest

#include "golay.h"

#include <stdio.h>

extern int golay_numones[];

const int num_eptn = 1+24+12*23+4*23*22;
int eptn[num_eptn];

void create_eptn(int* eptn);

int main(int argc, char** argv)
{
	create_eptn(eptn);

	// Try all possible messages with all possible error patterns of
	// weight at most three.
	for(int m = 0; m < (1 << 12); m++)
	{
		printf("m=%04x...\r", m);
		fflush(stdout);

		int c = golay_encode(m);

		for(int e = 0; e < num_eptn; e++)
		{
			int r = c ^ eptn[e];
			int d = golay_decode(r);

			if(d != m)
				printf("m=%04x, e=%08x failed\n", m, e);
		}
	}

	return 0;
}

// Create a table of all possible error patterns of three bits
void create_eptn(int* eptn)
{
	int k = 0;
	int e = 0;

	eptn[k++] = e;

	for(int e1 = 0; e1 < 24; e1++)
	{
		e = 0;
		e |= (1 << e1);
		eptn[k++] = e;
		for(int e2 = e1+1; e2 < 24; e2++)
		{
			e = 0;
			e |= (1 << e1);
			e |= (1 << e2);
			eptn[k++] = e;
			for(int e3 = e2+1; e3 < 24; e3++)
			{
				e = 0;
				e |= (1 << e1);
				e |= (1 << e2);
				e |= (1 << e3);
				eptn[k++] = e;
			}
		}
	}
}

