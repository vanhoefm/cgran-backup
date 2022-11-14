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
#ifndef INCLUDED_PRBLK_FRAMER_NONE_VBB_H
#define INCLUDED_PRBLK_FRAMER_NONE_VBB_H

#include <pr_framer_vbb.h>

class pr_framer_none_vbb;

typedef boost::shared_ptr<pr_framer_none_vbb> pr_framer_none_vbb_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Size of input and output vectors.
 */
pr_framer_none_vbb_sptr pr_make_framer_none_vbb(int frame_size);

/**
 * \brief Dummy framer.
 *
 * \ingroup sigblk
 * The input is copied verbatim to the output. No frame synchronization is 
 * possible.
 *
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[frame_size])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[frame_size])
 */
class pr_framer_none_vbb : public pr_framer_vbb
{
private:
	friend pr_framer_none_vbb_sptr pr_make_framer_none_vbb(int frame_size);

	pr_framer_none_vbb(int frame_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_framer_none_vbb();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);
};

#endif

