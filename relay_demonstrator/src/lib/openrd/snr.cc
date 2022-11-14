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

#include "openrd_debug.h"
#include <algorithm>
#include <math.h>

float snr_estimate(const gr_complex* data, int len)
{
	double psig;
	double pnoise;
	double snr;

	double m2 = 0.0;
	double m4 = 0.0;

	double sum_m2 = 0.0;
	double sum_m4 = 0.0;

	for(int k = 0; k < len; k++)
	{
		double n = std::norm(data[k]);

		sum_m2 += n;
		sum_m4 += n*n;
	}

	m2 = sum_m2 / len;
	m4 = sum_m4 / len;

	psig = sqrt(2 * m2*m2 - m4);
	pnoise = m2 - psig;

	if(pnoise > 0.0)
		snr = psig/pnoise;
	else
		snr = 0.0;

	return snr;
}

