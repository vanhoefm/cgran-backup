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

#include "modulation.h"

int modulation_bits_per_symbol(modulation_type modulation)
{
	switch(modulation)
	{
	case MODULATION_BPSK:
		return 1;
	case MODULATION_QPSK:
		return 2;
	default:
		throw "modulation_bits_per_symbol: invalid modulation type";
	}
}

void modulation_demod_bpsk(char* out, const gr_complex* in, int len)
{
	for(int k = 0; k < len; k++)
	{
		out[k] = (in[k].real() >= 0);
	}
}

void modulation_demod_qpsk(char* out, const gr_complex* in, int len)
{
	for(int k = 0; k < len; k++)
	{
	}
}

void modulation_softdemod_bpsk_v(float* out, const gr_complex* in, int len)
{
	for(int k = 0; k < len; k++)
	{
		out[k] = in[k].real();
	}
}

void modulation_softdemod_qpsk_v(float* out, const gr_complex* in, int len)
{
}

