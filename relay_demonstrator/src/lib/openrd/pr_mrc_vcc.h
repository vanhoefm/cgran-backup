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
#ifndef INCLUDED_PRBLK_MRC_VCC_H
#define INCLUDED_PRBLK_MRC_VCC_H

#include <gr_sync_block.h>

class pr_mrc_vcc;

typedef boost::shared_ptr<pr_mrc_vcc> pr_mrc_vcc_sptr;

/**
 * Public constructor.
 *
 * \param size Block size.
 */
pr_mrc_vcc_sptr pr_make_mrc_vcc(int size);

/**
 * \brief Maximum ratio combining.
 *
 * \ingroup sigblk
 * The class performs MRC on two inputs. Each input is scaled by 
 * sqrt(\p power), where \p power is taken from the header, and the results
 * are added together.
 *
 * Ports
 *  - Input 0: (<b>rxframe</b>, <b>complex</b>[size])
 *  - Input 1: (<b>rxframe</b>, <b>complex</b>[size])
 *  - Output 0: (<b>rxframe</b>, <b>complex</b>[size])
 */
class pr_mrc_vcc : public gr_sync_block
{
private:
	friend pr_mrc_vcc_sptr pr_make_mrc_vcc(int size);

	pr_mrc_vcc(int size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_mrc_vcc();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the block size.
	 */
	int size() const;
private:
	int d_size;
};

#endif

