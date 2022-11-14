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
#ifndef INCLUDED_PRBLK_CONST_MAPPER_VBC_H
#define INCLUDED_PRBLK_CONST_MAPPER_VBC_H

#include <gr_block.h>

#include "modulation.h"

class pr_const_mapper_vbc;

typedef boost::shared_ptr<pr_const_mapper_vbc> pr_const_mapper_vbc_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Number of bits in input vector
 * \param modulation the modulation type
 */
pr_const_mapper_vbc_sptr pr_make_const_mapper_vbc(int frame_size, modulation_type modulation);

/**
 * \brief Maps bits to symbols
 *
 * \ingroup sigblk
 * This block maps vectors of bits to vectors of symbols. Currently, only BPSK is implemented.
 *
 * Ports
 *  - Input 0: (<b>txmeta</b>, <b>char</b>[frame_size])
 *  - Output 0: <b>txmeta</b>, <b>complex</b>[frame_size])
 */
class pr_const_mapper_vbc : public gr_block
{
private:
	friend pr_const_mapper_vbc_sptr pr_make_const_mapper_vbc(int frame_size, modulation_type modulation);

	pr_const_mapper_vbc(int frame_size, modulation_type modulation);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_const_mapper_vbc();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * \return the modulation type
	 */
	modulation_type modulation() const;

private:
	int d_frame_size;
	unsigned int d_pkt;
	bool d_data_valid;
	modulation_type d_modulation;
};

#endif

