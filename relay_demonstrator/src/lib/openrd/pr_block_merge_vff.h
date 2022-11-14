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
#ifndef INCLUDED_PRBLK_BLOCK_MERGE_VFF_H
#define INCLUDED_PRBLK_BLOCK_MERGE_VFF_H

#include <gr_block.h>

class pr_block_merge_vff;

typedef boost::shared_ptr<pr_block_merge_vff> pr_block_merge_vff_sptr;

/**
 * Public constructor.
 *
 * \param insize length of input vectors
 * \param outsize length of output vectors
 */
pr_block_merge_vff_sptr pr_make_block_merge_vff(int insize, int outsize);

/**
 * \brief Merges partitions into packets
 *
 * \ingroup sigblk
 * The input vectors of size \p insize are merged into output vectors of size
 * \p outsize. Each output vector has a packet sequence number, and consists
 * of all the input vectors with the same packet sequence number. In the 
 * input, the frame sequence numbers are used to decide their places in the
 * output vectors. An output vector is produced each time the packet sequence
 * number of the input changes.
 *
 * Ports
 *  - Input 0: (<b>\ref rxframe</b>, <b>char</b>[insize])
 *  - Output 0: (<b>\ref rxmeta</b>, <b>char</b>[outsize])
 */
class pr_block_merge_vff : public gr_block
{
private:
	friend pr_block_merge_vff_sptr pr_make_block_merge_vff(int insize, int outsize);

	pr_block_merge_vff(int insize, int outsize);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_block_merge_vff();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Sets the range for the packet sequence numbers (to avoid errors when
	 * the number overflows).
	 *
	 * \param numpktseq number of packet sequence numbers to use
	 */
	void set_numpktseq(unsigned int numpktseq);

private:
	int d_insize;
	int d_outsize;
	unsigned int d_parts;
	unsigned int d_pkt;
	bool d_first_part;
	float* d_block;
	unsigned int d_numpktseq;
};

#endif

