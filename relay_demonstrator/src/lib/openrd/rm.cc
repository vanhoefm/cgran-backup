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

#include "rm.h"

void rm_encode(const rm_t* rm, char* m, char* e)
{
	for(int i = 0; i < rm->n; i++)
	{
		int t = 0;

		for(int j = 0; j < rm->k; j++)
			t += (m[j] & rm->G[j][i]);

		e[i] = (t & 0x01);
	}
}

void rm_decode(const rm_t* rm, char* x, char* d)
{
	char s[rm->n];
	char xv[rm->n];

	int di = rm->k;

	for(int i = 0; i < rm->n; i++)
		xv[i] = x[i];

	for(int ri = rm->r; ri >= 0; ri--)
	{
		int ncomb = (1 << (rm->m - ri));
		int nvar = (1 << ri);

		for(int i = 0; i < rm->n; i++)
			s[i] = 0;

		for(int pi = 0; pi < rm->mlvsize[ri]; pi++)
		{
			int v = 0;

			di--;

			for(int ci = 0; ci < ncomb; ci++)
			{
				int t = 0;

				for(int vi = 0; vi < nvar; vi++)
				{
					t += xv[rm->mlv[ri][pi][ci][vi]];
				}

				if(t & 1 == 1)
					v++;
			}
			if(v > (ncomb-v))
			{
				d[di] = 1;

				for(int i = 0; i < rm->n; i++)
					s[i] ^= rm->G[di][i];
			}
			else
				d[di] = 0;
		}

		for(int i = 0; i < rm->n; i++)
			xv[i] ^= s[i];
	}
}

