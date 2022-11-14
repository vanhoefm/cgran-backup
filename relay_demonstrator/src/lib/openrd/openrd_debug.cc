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

#include "openrd_debug.h"

#include <iostream>
#include <boost/thread.hpp>
#include <map>
#include "frametime.h"

using namespace std;

static boost::mutex print_mutex;
static boost::mutex work_mutex;

struct work_data
{
	std::string name;
	int id;
	int nout_requested;
	int nout_used;
	unsigned int now;
	map<int, int> nin_avail;
	map<int, int> nin_used;
};

static map<const void*,work_data*> work_list;

static void print_work_data(work_data* d);

void _work_enter(const gr_block* block, 
		int noutput_items, 
		const vector<int>& ninput_items, 
		const vector<const void*>& input_items, 
		const vector<void*>& output_items)
{
	work_data* d = new work_data();
	
	d->name = block->name();
	d->id = block->unique_id();
	d->nout_requested = noutput_items;
	d->now = frametime_now();

	for(unsigned int k = 0; k < ninput_items.size(); k++)
		d->nin_avail[k] = ninput_items[k];

	boost::mutex::scoped_lock l(work_mutex);

	if(work_list.find(block) != work_list.end())
	{
		cerr << "work_exit not called for block " << block->name() << endl;
		return;
	}

	work_list[block] = d;
}

void _work_used(const gr_block* block, int input, int nitems)
{
	boost::mutex::scoped_lock l(work_mutex);

	if(work_list.find(block) == work_list.end())
	{
		cerr << "work_used called for block " << block->name() << " before call to work_enter" << endl;
		return;
	}

	work_list[block]->nin_used[input] = nitems;
}

void _work_exit(const gr_block* block, int nitems)
{
	boost::mutex::scoped_lock l(work_mutex);

	map<const void*,work_data*>::iterator it(work_list.find(block));

	if(it == work_list.end())
	{
		cerr << "work_exit called for block " << block->name() << " before call to work_enter" << endl;
		return;
	}

	work_data* d = it->second;
	work_list.erase(it);
	l.unlock();

	d->nout_used = nitems;

	print_work_data(d);

	delete d;
}

static void print_work_data(work_data* d)
{
	boost::mutex::scoped_lock l(print_mutex);

	cerr << d->now << " ";
	cerr << d->name << "(" << d->id << "): ";
	cerr << d->nout_requested << "(" << d->nout_used << ")   ";

	for(unsigned int k = 0; k < d->nin_avail.size(); k++)
	{
		cerr << d->nin_avail[k] << "(" << d->nin_used[k] << ") ";
	}

	cerr << endl;
}

