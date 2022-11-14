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
#ifndef INCLUDED_PR_DATA_SINK_H
#define INCLUDED_PR_DATA_SINK_H

#include <gr_sync_block.h>

class pr_data_sink;

typedef boost::shared_ptr<pr_data_sink> pr_data_sink_sptr;

/**
 * \brief Base class for data sinks.
 *
 * \ingroup sigblk
 * The input is a vector of unpacked binary data (one bit per byte) of the
 * specified size. Only the lsb is used. Processing the data is done by the
 * subclasses.
 *
 * Ports
 *  - Input 0: (<b>rxmeta</b>, <b>char</b>[block_size])
 *  - Output 0: (<b>rxmeta</b>, <b>char</b>[block_size])
 */
class pr_data_sink : public gr_sync_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param block_size Block size in bits.
	 */
	pr_data_sink(int block_size);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~pr_data_sink();

	/**
	 * Returns the block size.
	 */
	int block_size() const;

	/**
	 * Sets number of packet sequence numbers. (Default is 2048.)
	 *
	 * \param numpktseq Argument (0 .. \p num_pkt_seq-1 will be used).
	 */
	void set_numpktseq(unsigned int numpktseq);

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

protected:
	/**
	 * Implemented in subclasses to process received data. Only the lsb is
	 * used.
	 *
	 * \param data Pointer to received data. block_size() bytes are guaranteed
	 * to be valid.
	 */
	virtual void handle_packet(const unsigned char* data) = 0;

private:
	int d_block_size;
	unsigned int d_numpktseq;
	unsigned int d_pkt_seq;
};

#endif

