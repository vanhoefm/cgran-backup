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
#ifndef INCLUDED_PRBLK_FRAME_CORRELATOR_SIMPLE_BB_H
#define INCLUDED_PRBLK_FRAME_CORRELATOR_SIMPLE_BB_H

#include "pr_frame_correlator_bb.h"
#include <vector>

class pr_frame_correlator_simple_bb;

typedef boost::shared_ptr<pr_frame_correlator_simple_bb> pr_frame_correlator_simple_bb_sptr;

/**
 * Public constructor.
 *
 * \param input_size Number of bits per symbol.
 * \param frame_size Number of bits per frame.
 * \param access_code Synchronization sequence.
 * \param nrequired Number of correct bits required for positive correlation.
 */
pr_frame_correlator_simple_bb_sptr pr_make_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired);

/**
 * \brief Correlator for simple frames.
 *
 * \ingroup sigblk
 * Correlator using a simple correlation sequence at the start of each
 * frame. The output is 1, when at least \p nrequired bits of the last input
 * bits are equal to \p access_code.
 *
 * Ports
 *  - Input 0: <b>char</b>[input_size]
 *  - Output 0: <b>char</b>
 */
class pr_frame_correlator_simple_bb : public pr_frame_correlator_bb
{
private:
	friend pr_frame_correlator_simple_bb_sptr pr_make_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired);

	pr_frame_correlator_simple_bb(int input_size, int frame_size, const std::vector<char>& access_code, int nrequired);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_frame_correlator_simple_bb();

protected:
	virtual int correlate(const char* data);

private:
	int d_cnt;
	std::vector<char> d_access_code;
	int d_nrequired;
};

#endif

