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
#ifndef INCLUDED_PRBLK_BLOCK_DECODER_NONE_VFB_H
#define INCLUDED_PRBLK_BLOCK_DECODER_NONE_VFB_H

#include <pr_block_decoder_vfb.h>

class pr_block_decoder_none_vfb;

typedef boost::shared_ptr<pr_block_decoder_none_vfb> pr_block_decoder_none_vfb_sptr;

/**
 * Public constructor.
 *
 * \param size size of input and output vectors
 */
pr_block_decoder_none_vfb_sptr pr_make_block_decoder_none_vfb(int size);

/**
 * \brief Dummy block decoder
 *
 * \ingroup sigblk
 * This block performs a hard decision of the input vectors, such that:
 *
 * out = (in >= 0.0)
 *
 * Ports
 *  - Input 0: (<b>\ref rxmeta</b>, <b>float</b>[size])
 *  - Output 0: (<b>\ref rxmeta</b>, <b>char</b>[size])
 */
class pr_block_decoder_none_vfb : public pr_block_decoder_vfb
{
private:
	friend pr_block_decoder_none_vfb_sptr pr_make_block_decoder_none_vfb(int size);

	pr_block_decoder_none_vfb(int size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_block_decoder_none_vfb();

protected:
	virtual int decode(char* dec, const float* rec);
};

#endif

