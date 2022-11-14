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
#ifndef INCLUDED_PRBLK_DATA_SOURCE_COUNTER_H
#define INCLUDED_PRBLK_DATA_SOURCE_COUNTER_H

#include "pr_data_source.h"

class pr_data_source_counter;

typedef boost::shared_ptr<pr_data_source_counter> pr_data_source_counter_sptr;

/**
 * Public constructor.
 *
 * \param block_size Block size in bits.
 */
pr_data_source_counter_sptr pr_make_data_source_counter(int block_size);

/**
 * \brief Deterministic data generator.
 *
 * \ingroup sigblk
 * Each block consists of four-bit values (msb first), incremented from 0 to 
 * 15 and then repeated. All generated blocks are identical. Thus, each block
 * contains the sequence:
 *
 * 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0 1 ...
 *
 * Ports
 *  - Output 0: <b>char</b>[block_size]
 */
class pr_data_source_counter : public pr_data_source
{
private:
	friend pr_data_source_counter_sptr pr_make_data_source_counter(int block_size);

	pr_data_source_counter(int block_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_data_source_counter();

protected:
	virtual void fill_packet(unsigned char* data, unsigned char* valid);
};

#endif

