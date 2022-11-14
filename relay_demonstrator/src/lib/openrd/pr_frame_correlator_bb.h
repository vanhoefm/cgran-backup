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
#ifndef INCLUDED_PRBLK_FRAME_CORRELATOR_BB_H
#define INCLUDED_PRBLK_FRAME_CORRELATOR_BB_H

#include <gr_sync_block.h>

class pr_frame_correlator_bb;

typedef boost::shared_ptr<pr_frame_correlator_bb> pr_frame_correlator_bb_sptr;

/**
 * \brief Base class for frame correlators.
 *
 * \ingroup sigblk
 * When a valid frame is found, the block outputs the frame type, otherwise
 * 0. In order to preserve causality, the output is delayed by the number
 * of symbols that constitute a frame.
 *
 * Ports
 *  - Input 0: <b>char</b>[input_size]
 *  - Output 0: <b>char</b>
 */
class pr_frame_correlator_bb : public gr_sync_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param input_size Number of bits per symbol.
	 * \param frame_size Size of frame in bits.
	 *
	 * The behaviour is undefined if \p frame_size/\p input_size is not an 
	 * integer.
	 */
	pr_frame_correlator_bb(int input_size, int frame_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_frame_correlator_bb();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the number of bits per symbol (length of input vector).
	 */
	int input_size() const;

	/**
	 * Returns the number of bits per frame.
	 */
	int frame_size() const;

	/**
	 * Returns frame_size()/input_size().
	 */
	int delay() const;

protected:
	/**
	 * Correlation function, reimplemented in subclasses. The function
	 * returns the frame type if \p data points to a valid frame, otherwise
	 * 0.
	 */
	virtual int correlate(const char* data) = 0;

private:
	int d_input_size;
	int d_frame_size;
};

#endif

