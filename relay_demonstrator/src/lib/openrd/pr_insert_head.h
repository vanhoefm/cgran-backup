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
#ifndef INCLUDED_PRBLK_INSERT_HEAD_H
#define INCLUDED_PRBLK_INSERT_HEAD_H

#include <gr_block.h>

class pr_insert_head;

typedef boost::shared_ptr<pr_insert_head> pr_insert_head_sptr;

/**
 * Public constructor.
 *
 * \param sizeof_stream_item Size (in bytes) of elements.
 * \param nitems Number of items to insert.
 */
pr_insert_head_sptr pr_make_insert_head(int sizeof_stream_item, int nitems);

/**
 * \brief Inserts null elements at head of stream.
 *
 * \ingroup primblk
 * The output is \p nitems of null data, followed by the input passed through unchanged.
 *
 * Ports
 *  - Input 0: <b>char</b>[sizeof_stream_item]
 *  - Output 0: <b>char</b>[sizeof_stream_item]
 */
class pr_insert_head : public gr_block
{
private:
	friend pr_insert_head_sptr pr_make_insert_head(int sizeof_stream_item, int nitems);

	pr_insert_head(int sizeof_stream_item, int nitems);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_insert_head();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	virtual void forecast(int noutput_items, gr_vector_int& ninput_items_required);

private:
	int d_sizeof_stream_item;
	int d_nitems;
	int d_current;
};

#endif

