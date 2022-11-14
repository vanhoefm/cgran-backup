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
#ifndef INCLUDED_PRBLK_FRAMER_SIMPLE_VBB_H
#define INCLUDED_PRBLK_FRAMER_SIMPLE_VBB_H

#include <pr_framer_vbb.h>
#include <vector>

class pr_framer_simple_vbb;

typedef boost::shared_ptr<pr_framer_simple_vbb> pr_framer_simple_vbb_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Length of frames.
 * \param access_code Frame synchronization sequence.
 */
pr_framer_simple_vbb_sptr pr_make_framer_simple_vbb(int frame_size, const std::vector<char>& access_code);

/**
 * \brief Simple framer with synchronization sequence at start of each frame.
 *
 * \ingroup sigblk
 * The size of the input vectors is defined as \p data_size = \p frame_size -
 * \p access_code.size().
 *
 * Since the \p pkt_seq and \p frame_seq fields of the header are not encoded
 * in the frame structure in any way, the simple framer does not have a way
 * of detecting lost frames. It is intended mainly for testing of low-level
 * communication functionalities such as modulation and frequency and symbol
 * synchronization. It is not useful for transmission of general data.
 *
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[data_size])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[frame_size])
 */
class pr_framer_simple_vbb : public pr_framer_vbb
{
private:
	friend pr_framer_simple_vbb_sptr pr_make_framer_simple_vbb(int frame_size, const std::vector<char>& access_code);

	pr_framer_simple_vbb(int frame_size, const std::vector<char>& access_code);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_framer_simple_vbb();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

private:
	std::vector<char> d_access_code;
};

#endif

