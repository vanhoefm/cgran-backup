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
#ifndef INCLUDED_PRBLK_FRAME_CORRELATOR_NONE_BB_H
#define INCLUDED_PRBLK_FRAME_CORRELATOR_NONE_BB_H

#include "pr_frame_correlator_bb.h"

class pr_frame_correlator_none_bb;

typedef boost::shared_ptr<pr_frame_correlator_none_bb> pr_frame_correlator_none_bb_sptr;

/**
 * Public constructor.
 *
 * \param input_size Number of bits per symbol.
 * \param frame_size Number of bits per frame.
 */
pr_frame_correlator_none_bb_sptr pr_make_frame_correlator_none_bb(int input_size, int frame_size);

/**
 * \brief Dummy frame correlator.
 *
 * \ingroup sigblk
 * The class implements a dummy frame correlator, returning 1 every 
 * frame_size()/input_size() elements.
 */
class pr_frame_correlator_none_bb : public pr_frame_correlator_bb
{
private:
	friend pr_frame_correlator_none_bb_sptr pr_make_frame_correlator_none_bb(int input_size, int frame_size);

	pr_frame_correlator_none_bb(int input_size, int frame_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_frame_correlator_none_bb();

protected:
	virtual int correlate(const char* data);

private:
	int d_cnt;
};

#endif

