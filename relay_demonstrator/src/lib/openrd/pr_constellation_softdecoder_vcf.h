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
#ifndef INCLUDED_PRBLK_CONSTELLATION_SOFTDECODER_VCF_H
#define INCLUDED_PRBLK_CONSTELLATION_SOFTDECODER_VCF_H

#include <gr_sync_block.h>

#include "modulation.h"

class pr_constellation_softdecoder_vcf;

typedef boost::shared_ptr<pr_constellation_softdecoder_vcf> pr_constellation_softdecoder_vcf_sptr;

/**
 * Public constructor.
 *
 * \param modulation Modulation type
 * \param size Number of complex symbols in input vector
 */
pr_constellation_softdecoder_vcf_sptr pr_make_constellation_softdecoder_vcf(modulation_type modulation, int size);

/**
 * \brief Constellation decoder with soft output.
 *
 * \ingroup sigblk
 * The block decodes complex symbols to soft measurements for each received bit.
 *
 * Ports
 *  - Input 0: (<b>rxframe</b>, <b>complex</b>[size])
 *  - Output 0: (<b>rxframe</b>, <b>float</b>[nbits])
 *
 *  where \p nbits = \p bits_per_symbol * \p size
 */
class pr_constellation_softdecoder_vcf : public gr_sync_block
{
private:
	friend pr_constellation_softdecoder_vcf_sptr pr_make_constellation_softdecoder_vcf(modulation_type modulation, int size);

	pr_constellation_softdecoder_vcf(modulation_type modulation, int size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_constellation_softdecoder_vcf();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * \return the modulation type
	 */
	modulation_type modulation() const;

	/**
	 * \return the number of bits per symbol
	 */
	int symbol_bits() const;

	/**
	 * \return the number of symbols in the input vector
	 */
	int size() const;

private:
	modulation_type d_modulation;
	int d_size;
};

#endif

