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
#ifndef INCLUDED_PRBLK_PVEC_EXTRACT_H
#define INCLUDED_PRBLK_PVEC_EXTRACT_H

#include <gr_sync_block.h>

class pr_pvec_extract;

typedef boost::shared_ptr<pr_pvec_extract> pr_pvec_extract_sptr;

/**
 * Public constructor.
 *
 * \param insize Size of input elements.
 * \param offset Offset of element to extract.
 * \param outsize Size of element to extract.
 */
pr_pvec_extract_sptr pr_make_pvec_extract(int insize, int offset, int outsize);

/**
 * \brief Extracts elements in a stream of padded vectors.
 *
 * \ingroup primblk
 * For each element of \p insize bytes, an element of \p outsize bytes is
 * produced. The element is extracted from the offset \p offset in the input.
 *
 * Ports
 *  - Input 0: <b>char</b>[insize] (padded to a power of two)
 *  - Output 0: <b>char</b>[outsize] (padded to a power of two)
 */
class pr_pvec_extract : public gr_sync_block
{
private:
	friend pr_pvec_extract_sptr pr_make_pvec_extract(int insize, int offset, int outsize);

	pr_pvec_extract(int insize, int offset, int outsize);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_pvec_extract();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the input element size.
	 */
	int insize() const;

	/**
	 * Returns the extraction offset.
	 */
	int offset() const;

	/**
	 * Returns the output element size.
	 */
	int outsize() const;

private:
	int d_insize;
	int d_offset;
	int d_outsize;
};

#endif

