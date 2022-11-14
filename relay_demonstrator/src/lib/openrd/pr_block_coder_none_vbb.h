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
#ifndef INCLUDED_PRBLK_BLOCK_CODER_NONE_VBB_H
#define INCLUDED_PRBLK_BLOCK_CODER_NONE_VBB_H

#include <pr_block_coder_vbb.h>

class pr_block_coder_none_vbb;

typedef boost::shared_ptr<pr_block_coder_none_vbb> pr_block_coder_none_vbb_sptr;

/**
 * Public constructor.
 *
 * \param size Size of input and output vectors.
 */
pr_block_coder_none_vbb_sptr pr_make_block_coder_none_vbb(int size);

/**
 * \brief Dummy block coder with no parity bits.
 *
 * \ingroup sigblk
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[size])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[size])
 */
class pr_block_coder_none_vbb : public pr_block_coder_vbb
{
private:
	friend pr_block_coder_none_vbb_sptr pr_make_block_coder_none_vbb(int size);

	pr_block_coder_none_vbb(int size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_block_coder_none_vbb();

protected:
	virtual void encode(char* codeword, const char* src);
};

#endif

