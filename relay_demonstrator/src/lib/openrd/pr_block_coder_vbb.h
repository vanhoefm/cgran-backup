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
#ifndef INCLUDED_PRBLK_BLOCK_CODER_VBB_H
#define INCLUDED_PRBLK_BLOCK_CODER_VBB_H

#include <gr_sync_block.h>

class pr_block_coder_vbb;

typedef boost::shared_ptr<pr_block_coder_vbb> pr_block_coder_vbb_sptr;

/**
 * \brief Base class for block coders.
 *
 * \ingroup sigblk
 * The class codes vectors of information bits to vectors of codewords.
 *
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[information_size])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[codeword_size])
 */
class pr_block_coder_vbb : public gr_sync_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param information_size Number of information bits
	 * \param codeword_size Number of codeword bits
	 */
	pr_block_coder_vbb(int information_size, int codeword_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_block_coder_vbb();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the number of information bits.
	 */
	int information_size() const;

	/**
	 * Returns the number of codeword bits.
	 */
	int codeword_size() const;

protected:
	/**
	 * Encodes a vector of information bits to a codeword. Reimplemented in
	 * subclasses.
	 *
	 * \param codeword Pointer to codeword.
	 * \param src Pointer to vector of information bits.
	 */
	virtual void encode(char* codeword, const char* src) = 0;

private:
	int d_information_size;
	int d_codeword_size;
};

#endif

