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
#ifndef INCLUDED_PRBLK_DAF_LOGIC_VBB_H
#define INCLUDED_PRBLK_DAF_LOGIC_VBB_H

#include <gr_block.h>

class pr_daf_logic_vbb;

typedef boost::shared_ptr<pr_daf_logic_vbb> pr_daf_logic_vbb_sptr;

pr_daf_logic_vbb_sptr pr_make_daf_logic_vbb(int block_size);

class pr_daf_logic_vbb : public gr_block
{
private:
	friend pr_daf_logic_vbb_sptr pr_make_daf_logic_vbb(int block_size);

	pr_daf_logic_vbb(int block_size);

public:
	virtual ~pr_daf_logic_vbb();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

private:
	int d_block_size;
	unsigned int d_pkt;
	unsigned int d_frame;
	bool d_data_valid;
};

#endif

