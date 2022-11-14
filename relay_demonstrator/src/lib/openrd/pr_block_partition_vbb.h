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
#ifndef INCLUDED_PRBLK_BLOCK_PARTITION_VBB_H
#define INCLUDED_PRBLK_BLOCK_PARTITION_VBB_H

#include <gr_sync_interpolator.h>

class pr_block_partition_vbb;

typedef boost::shared_ptr<pr_block_partition_vbb> pr_block_partition_vbb_sptr;

/**
 * Public constructor.
 *
 * \param insize The length of the input vectors.
 * \param outsize The length of the output vectors.
 */
pr_block_partition_vbb_sptr pr_make_block_partition_vbb(int insize, int outsize);

/**
 * \brief Partitions packets into partitions.
 *
 * \ingroup sigblk
 * Each input vector of size \p insize is partitioned into a number of output
 * vectors of size \p outsize. The behaviour is undefined if \p insize/
 * \p outsize is not an integer multiple.
 *
 * The \p pkt_seq field of the header is retained, and the \p frame_seq field
 * of the input is ignored. In the output, each partition of a packet is 
 * numbered sequentially in the \p frame_seq field.
 *
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[insize])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[outsize])
 */
class pr_block_partition_vbb : public gr_sync_interpolator
{
private:
	friend pr_block_partition_vbb_sptr pr_make_block_partition_vbb(int insize, int outsize);
	pr_block_partition_vbb(int insize, int outsize);

public:
	/**
	 * Public destructor.
	 */
	~pr_block_partition_vbb();
	virtual int work(int noutput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items);

private:
	int d_insize;
	int d_outsize;
	int d_factor;
};

#endif

