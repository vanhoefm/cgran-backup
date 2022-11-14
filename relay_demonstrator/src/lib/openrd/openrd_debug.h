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
#ifndef INCLUDED_OPENRD_DEBUG_H
#define INCLUDED_OPENRD_DEBUG_H

#include <vector>
#include <gr_block.h>

using namespace std;

/* Defining this will cause all blocks to print out traces of the arguments
 * and return values for all the work/general_work calls. */
//#define OPENRD_DEBUG

#ifdef OPENRD_DEBUG
#define work_enter(a,b,c,d,e) _work_enter(a,b,c,d,e)
#define work_used(a,b,c) _work_used(a,b,c)
#define work_exit(a,b) _work_exit(a,b)
#else
#define work_enter(a,b,c,d,e)
#define work_used(a,b,c)
#define work_exit(a,b)
#endif

void _work_enter(const gr_block* block, 
		int noutput_items, 
		const vector<int>& ninput_items, 
		const vector<const void*>& input_items, 
		const vector<void*>& output_items);

extern inline void _work_enter(const gr_block* block, 
		int noutput_items, 
		int ninput_items, 
		const vector<const void*>& input_items, 
		const vector<void*>& output_items)
{ _work_enter(block, noutput_items, vector<int>(), input_items, output_items); }

void _work_used(const gr_block* block, int input, int nitems);
void _work_exit(const gr_block* block, int nitems);

#endif

