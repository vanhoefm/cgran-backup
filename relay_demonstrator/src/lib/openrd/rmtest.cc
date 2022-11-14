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
// $ g++ rmtest.cc rm_2_6.cc rm.cc -o rmtest

#include "rm_2_6.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void generate_message(char* m);
void scramble(char* e, char* x, int n);
void print_vector(const char* x, int k);

int main(int argc, char** argv)
{
	char m[22];
	char e[64];
	char x[64];
	char d[22];
	int n = 0;

	while(true)
	{
		if(n % 1000 == 0)
		{
			printf("%d\r", n);
			fflush(stdout);
		}

		generate_message(m);
		rm_2_6_encode(m, e);
		scramble(e, x, 7);
		rm_2_6_decode(x, d);
		if(memcmp(m, d, 22) != 0)
		{
			printf("F: m=");
			print_vector(m,22);
			printf(", e=");
			print_vector(e,64);
			printf("\n");
			printf("   d=");
			print_vector(d,22);
			printf(", x=");
			print_vector(x,64);
			printf("\n");
		}
		n++;
	}

	return 0;
}

void generate_message(char* m)
{
	int r = rand();
	for(int i = 0; i < 22; i++)
	{
		m[i] = (r & 0x01);
		r >>= 1;
	}
}

void scramble(char* e, char* x, int n)
{
	for(int i = 0; i < 64; i++)
		x[i] = e[i];

	for(int i = 0; i < n; i++)
		x[rand()&63] ^= 1;
}

void print_vector(const char* x, int k)
{
	for(int i = 0; i < k; i++)
		printf("%d", x[i]);
}

