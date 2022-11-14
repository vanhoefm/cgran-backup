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
#ifndef INCLUDED_PRBLK_FRAMER_VBB_H
#define INCLUDED_PRBLK_FRAMER_VBB_H

#include <gr_block.h>

class pr_framer_vbb;

typedef boost::shared_ptr<pr_framer_vbb> pr_framer_vbb_sptr;

/**
 * \brief Base class for framers.
 *
 * \ingroup sigblk
 * The purpose of the framers is to add a frame structure with 
 * synchronization and training fields to the data. The input-to-output
 * rate is not fixed, so the implementations may add extra frames to the
 * output.
 *
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[data_size])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[frame_size])
 */
class pr_framer_vbb : public gr_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param data_size Length of input vector.
	 * \param frame_size Length of output vector.
	 */
	pr_framer_vbb(int data_size, int frame_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_framer_vbb();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items) = 0;

	/**
	 * Returns the length of the input vectors.
	 */
	int data_size() const;

	/**
	 * Returns the length of the output vectors.
	 */
	int frame_size() const;

private:
	int d_data_size;
	int d_frame_size;
};

#endif

