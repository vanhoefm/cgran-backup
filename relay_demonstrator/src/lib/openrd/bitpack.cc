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

#include "bitpack.h"

void bitpack_unpack(const unsigned char* src, unsigned char* dest, int nbytes)
{
	for(int i = 0; i < nbytes; i++)
	{
		unsigned char d = *src;

		for(int j = 0; j < 8; j++)
		{
			*dest = (d & 0x80) ? 1 : 0;
			d <<= 1;
			dest++;
		}
		src++;
	}
}

void bitpack_pack(const unsigned char* src, unsigned char* dest, int nbytes)
{
	for(int i = 0; i < nbytes; i++)
	{
		unsigned char d = 0;

		for(int j = 0; j < 8; j++)
		{
			d = (d << 1) + *src;
			src++;
		}
		*dest = d;
		dest++;
	}
}

