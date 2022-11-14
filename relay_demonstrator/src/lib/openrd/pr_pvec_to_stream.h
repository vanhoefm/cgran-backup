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
#ifndef INCLUDED_PRBLK_PVEC_TO_STREAM_H
#define INCLUDED_PRBLK_PVEC_TO_STREAM_H

#include <gr_sync_interpolator.h>

class pr_pvec_to_stream;

typedef boost::shared_ptr<pr_pvec_to_stream> pr_pvec_to_stream_sptr;

/**
 * Public constructor.
 *
 * \param item_size Size of elements (in bytes).
 * \param nitems Number of elements in input vector.
 */
pr_pvec_to_stream_sptr pr_make_pvec_to_stream(int item_size, int nitems);

/**
 * \brief Converts a stream of padded vectors to a stream of items.
 *
 * \ingroup primblk
 * For each input element, \p nitems items are produced on the output.
 *
 * Ports
 *  - Input 0: <b>char</b>[item_size*nitems] (padded to a power of two)
 *  - Output 0: <b>char</b>[item_size]
 */
class pr_pvec_to_stream : public gr_sync_interpolator
{
private:
	friend pr_pvec_to_stream_sptr pr_make_pvec_to_stream(int item_size, int nitems);

	pr_pvec_to_stream(int item_size, int nitems);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_pvec_to_stream();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the item size.
	 */
	int item_size() const;

	/**
	 * Returns the number of items in each input element.
	 */
	int nitems() const;

	/**
	 * Returns the actual allocation size of the input elements (a power of two).
	 */
	int alloc_size() const;

private:
	int d_item_size;
	int d_nitems;
	int d_alloc_size;
	int d_block_size;
	int d_skip_size;
};

#endif

