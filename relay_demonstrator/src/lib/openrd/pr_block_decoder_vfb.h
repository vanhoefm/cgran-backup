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
#ifndef INCLUDED_PRBLK_BLOCK_DECODER_VFB_H
#define INCLUDED_PRBLK_BLOCK_DECODER_VFB_H

#include <gr_sync_block.h>

class pr_block_decoder_vfb;

typedef boost::shared_ptr<pr_block_decoder_vfb> pr_block_decoder_vfb_sptr;

/**
 * \brief Base class for block decoders.
 *
 * \ingroup sigblk
 * This block performs soft decoding of data blocks.
 *
 * Ports
 *  - Input 0: (<b>\ref rxmeta</b>, <b>float</b>[codeword_size])
 *  - Output 0: (<b>\ref rxmeta</b>, <b>char</b>[information_size])
 */
class pr_block_decoder_vfb : public gr_sync_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param information_size number of information bits
	 * \param codeword_size number of codeword bits
	 */
	pr_block_decoder_vfb(int information_size, int codeword_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_block_decoder_vfb();

	/**
	 * \returns the number of information bits
	 */
	int information_size() const;

	/**
	 * \returns the number of codeword bits
	 */
	int codeword_size() const;

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

protected:
	/**
	 * Called for each input vector of <b>float</b> values. Must be 
	 * implemented in subclasses.
	 *
	 * \param rec vector of codeword_size() soft bits
	 * \param dec vector of information_size() hard bits
	 */
	virtual int decode(char* dec, const float* rec) = 0;

private:
	int d_information_size;
	int d_codeword_size;
};

#endif

