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
#ifndef INCLUDED_PRBLK_CONSTELLATION_DECODER_CB_H
#define INCLUDED_PRBLK_CONSTELLATION_DECODER_CB_H

#include <gr_sync_block.h>

#include "modulation.h"

class pr_constellation_decoder_cb;

typedef boost::shared_ptr<pr_constellation_decoder_cb> pr_constellation_decoder_cb_sptr;

/**
 * Public constructor.
 *
 * \param modulation Modulation type
 */
pr_constellation_decoder_cb_sptr pr_make_constellation_decoder_cb(modulation_type modulation);

/**
 * \brief Constellation decoder.
 *
 * \ingroup sigblk
 * The block decodes complex symbols to bits.
 *
 * Ports
 *  - Input 0: <b>complex</b>
 *  - Output 0: <b>char</b> (for BPSK)
 */
class pr_constellation_decoder_cb : public gr_sync_block
{
private:
	friend pr_constellation_decoder_cb_sptr pr_make_constellation_decoder_cb(modulation_type modulation);

	pr_constellation_decoder_cb(modulation_type modulation);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_constellation_decoder_cb();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Returns the modulation type.
	 */
	modulation_type modulation() const;

	/**
	 * Returns the number of bits per symbols (length of output vector):
	 * 
	 *  - BPSK: 1
	 */
	int symbol_bits() const;

private:
	modulation_type d_modulation;
};

#endif

