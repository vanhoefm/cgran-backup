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
#ifndef INCLUDED_PRBLK_FRAME_SYNC_BB_H
#define INCLUDED_PRBLK_FRAME_SYNC_BB_H

#include <gr_block.h>
#include <gr_complex.h>

class pr_frame_sync_bb;
typedef boost::shared_ptr<pr_frame_sync_bb> pr_frame_sync_bb_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Number of symbols per frame.
 */
pr_frame_sync_bb_sptr pr_make_frame_sync_bb(int frame_size);

/**
 * \brief Vectorizes a stream of symbols on frame boundaries
 *
 * \ingroup sigblk
 * The block has two inputs: one symbol input and one sync input. When the 
 * sync input is non-zero, \p frame_size symbols are read from the input and
 * output as a vector. Also, the \p stamp in the header is set to the current
 * time, and the \p frame_type is set to the sync input.
 *
 * Ports
 *  - Input 0: <b>char</b>
 *  - Output 0: (<b>rxframe</b>, <b>char</b>[frame_size])
 */
class pr_frame_sync_bb : public gr_block
{
private:
	friend pr_frame_sync_bb_sptr pr_make_frame_sync_bb(int frame_size);

	pr_frame_sync_bb(int frame_size);

public:
	/**
	 * Public destructor.
	 */
	~pr_frame_sync_bb();

	/**
	 * Returns the number of symbols per frame.
	 */
	int frame_size() const;

	int general_work(int noutput_items,	gr_vector_int& ninput_items, gr_vector_const_void_star& input_items, gr_vector_void_star& output_items);

private:
	enum state{STATE_NOSYNC, STATE_SYNC};

	state d_state;
	unsigned int d_frame_type;
	int	d_frame_size; // length of frame
	int	d_cnt; // how many so far
	char* d_frame; // contains the current frame

	void initialize();
};

#endif /* INCLUDED_PRBLK_FRAME_SYNC_BB_H */

