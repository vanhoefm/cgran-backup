/* -*- c++ -*- */
/*
 * Copyright 2004,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
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

#include "dabp_galois.h"
#include <cassert>

const int dabp_galois::PRIMITIVES[MAX_M]={03,07,013,023,045,0103,0211,0435,01021,02011};

dabp_galois::dabp_galois(int m)
{
	assert(m>=1 && m<=MAX_M);
	d_m=m;
	d_q=1<<m;
	d_p=PRIMITIVES[m-1];
	init_lut();
}

dabp_galois::~dabp_galois()
{
}

int dabp_galois::add_power(int a, int b)
{
	return lut_poly2power[add_poly(lut_power2poly[a],lut_power2poly[b])];
}

int dabp_galois::multiply_poly(int a, int b)
{
	return lut_power2poly[multiply_power(lut_poly2power[a],lut_poly2power[b])];
}

int dabp_galois::divide_power(int a, int b)
{
	assert(b!=0);
	return (a==0) ? 0 : round_mod(a-b,d_q-1)+1;
}

int dabp_galois::divide_poly(int a, int b)
{
	return lut_power2poly[divide_power(lut_poly2power[a],lut_poly2power[b])];
}

int dabp_galois::pow_poly(int a, int n)
{
	return lut_power2poly[pow_power(lut_poly2power[a],n)];
}

void dabp_galois::poly2tuple(int a, unsigned char tuple[])
{
	for(int i=0;i<d_m;i++,a>>=1)
		tuple[i]=(unsigned char)(a&1);
}

void dabp_galois::init_lut()
{
	int i;
	// zero element
	lut_power2poly[0]=0;
	lut_poly2power[0]=0;
	// from alpha^0 to alpha^(m-1)
	for(i=0;i<d_m;i++){
		// for alpha^i
		lut_power2poly[i+1]=1<<i;
		lut_poly2power[lut_power2poly[i+1]]=i+1;
	}
	if(d_m==1)
		return;

	int pm=d_p-d_q;
	// alpha^m=p(m)alpha^(m-1)+...+p(1)alpha+1
	lut_power2poly[d_m+1]=pm;
	lut_poly2power[pm]=d_m+1;
	for(i=d_m+2;i<d_q;i++){
		lut_power2poly[i]=(lut_power2poly[i-1]<<1)&(d_q-1);
		if(lut_power2poly[i-1]&(d_q>>1)){
			lut_power2poly[i]^=pm;
		}
		lut_poly2power[lut_power2poly[i]]=i;
	}
}
