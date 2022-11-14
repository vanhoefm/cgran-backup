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
#ifndef INCLUDED_PRBLK_DEFRAMER_SIMPLE_VCC_H
#define INCLUDED_PRBLK_DEFRAMER_SIMPLE_VCC_H

#include <pr_deframer_vcc.h>
#include <vector>

class pr_deframer_simple_vcc;

typedef boost::shared_ptr<pr_deframer_simple_vcc> pr_deframer_simple_vcc_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Size of input.
 * \param access_code Correlation sequence expected at start of each frame.
 */
pr_deframer_simple_vcc_sptr pr_make_deframer_simple_vcc(int frame_size, const std::vector<char>& access_code);

/**
 * \brief Deframer for simple frames.
 *
 * \ingroup sigblk
 * The correlation sequence is removed from each frame. In the header,
 * \p power is calculated, \p pkt_seq is set to an unbounded counter, 
 * and \p frame_seq is set to 0.
 *
 * Ports
 *  - Input 0: (<b>rxframe</b>, <b>complex</b>[frame_size])
 *  - Output 0: (<b>rxframe</b>, <b>complex</b>[data_size])
 *
 * \p data_size = \p frame_size - \p access_code.size()
 */
class pr_deframer_simple_vcc : public pr_deframer_vcc
{
private:
	friend pr_deframer_simple_vcc_sptr pr_make_deframer_simple_vcc(int frame_size, const std::vector<char>& access_code);

	pr_deframer_simple_vcc(int frame_size, const std::vector<char>& access_code);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_deframer_simple_vcc();

protected:
	virtual bool deframe(const rxframe& inmeta, const gr_complex* in,
			rxframe& outmeta, gr_complex* out);

private:
	int d_pkt;
	std::vector<char> d_access_code;
};

#endif

