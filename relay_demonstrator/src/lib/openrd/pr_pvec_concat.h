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
#ifndef INCLUDED_PRBLK_PVEC_CONCAT_H
#define INCLUDED_PRBLK_PVEC_CONCAT_H

#include <gr_sync_block.h>
#include <vector>

class pr_pvec_concat;

typedef boost::shared_ptr<pr_pvec_concat> pr_pvec_concat_sptr;

/**
 * Public constructor.
 *
 * \param sizes Vector of sizes of input elements.
 */
pr_pvec_concat_sptr pr_make_pvec_concat(const std::vector<int>& sizes);

/**
 * \brief Concatenates several padded vectors to one.
 *
 * \ingroup primblk
 * Ports
 *  - Input \p x: <b>char</b>[sizes[\p x]] (padded to a power of two)
 *  - Output 0: <b>char</b>[outsize] (padded to a power of two)
 *
 *  where \p outsize is the sum of \p sizes.
 */
class pr_pvec_concat : public gr_sync_block
{
private:
	friend pr_pvec_concat_sptr pr_make_pvec_concat(const std::vector<int>& sizes);

	pr_pvec_concat(const std::vector<int>& sizes);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_pvec_concat();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the sizes of the input elements.
	 */
	const std::vector<int>& sizes() const;

	/**
	 * Returns the size of the output.
	 */
	int outsize() const;

private:
	std::vector<int> d_sizes;
	int d_outsize;
};

#endif

