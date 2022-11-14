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
#ifndef INCLUDED_PRBLK_ANALYZER_BER_VB_H
#define INCLUDED_PRBLK_ANALYZER_BER_VB_H

#include <vector>

#include <pr_analyzer_vb.h>
#include <boost/thread/mutex.hpp>

class pr_analyzer_ber_vb;

typedef boost::shared_ptr<pr_analyzer_ber_vb> pr_analyzer_ber_vb_sptr;

/**
 * Public constructor.
 *
 * \param block_size size of input vectors
 * \param avg_blocks number of blocks to average over
 */
pr_analyzer_ber_vb_sptr pr_make_analyzer_ber_vb(int block_size, int avg_blocks);

/**
 * \brief Bit error rate analyzer for data blocks
 *
 * \ingroup sigblk
 * This block accepts two vectors as inputs and computes the bit error rate
 * and block error rate between them. The measurements are averaged over
 * a specified number of blocks, and the are available through the accessor
 * function ber() and bler().
 *
 * Ports
 *  - Input 0: (<b>txmeta</b>, <b>char</b>[block_size])
 *  - Input 1: (<b>rxmeta</b>, <b>char</b>[block_size])
 */
class pr_analyzer_ber_vb : public pr_analyzer_vb
{
private:
	friend pr_analyzer_ber_vb_sptr pr_make_analyzer_ber_vb(int block_size, int avg_blocks);

	pr_analyzer_ber_vb(int block_size, int avg_blocks);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_analyzer_ber_vb();

	/**
	 * \return vector of bit error rate measurements
	 */
	std::vector<float> ber();

	/**
	 * \return vector of block error rate measurements
	 */
	std::vector<float> bler();

protected:
	virtual void analyze(const txmeta& refmeta, const char* ref,
			const rxmeta& recmeta, const char* rec);

private:
	int d_block_size;
	int d_avg_blocks;
	int d_bits;
	int d_correctbits;
	int d_packets;
	int d_correctpackets;

	boost::mutex d_ber_lock; // Protects d_ber and d_bler
	std::vector<float> d_ber;
	std::vector<float> d_bler;
};

#endif
