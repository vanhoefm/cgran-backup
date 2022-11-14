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
#ifndef INCLUDED_PRBLK_DATA_SOURCE_H
#define INCLUDED_PRBLK_DATA_SOURCE_H

#include <gr_sync_block.h>

class pr_data_source;

typedef boost::shared_ptr<pr_data_source> pr_data_source_sptr;

/**
 * \brief Base class for data sources.
 *
 * \ingroup sigblk
 * The output is a vector of unpacked binary data (one bit per byte) of the
 * specified size. Only the lsb is used. Filling the vectors with data is
 * performed by the subclasses.
 *
 * Ports
 *  - Output 0: (<b>txmeta</b>, <b>char</b>[block_size])
 */
class pr_data_source : public gr_sync_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param block_size Block size in bits.
	 */
	pr_data_source(int block_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_data_source();

	/**
	 * Returns the block size.
	 */
	int block_size() const;

	/**
	 * Sets number of packet sequence numbers. (Default is 2048.)
	 *
	 * \param numpktseq Argument (0 .. \p num_pkt_seq-1 will be used).
	 */
	void set_numpktseq(int numpktseq);

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

protected:
	/**
	 * Implemented in subclasses to fill output vectors with data. Only lsb is 
	 * used.
	 *
	 * \param data Pointer to output vector. block_size() bytes are guaranteed 
	 * to be allocated.
	 * \param valid Corresponds to the data_valid field in the header. Should
	 * be set to 0 or 1 by the subclasses
	 */
	virtual void fill_packet(unsigned char* data, unsigned char* valid) = 0;

private:
	int d_block_size;
	int d_numpktseq;
	int d_pkt_seq;
};

#endif

