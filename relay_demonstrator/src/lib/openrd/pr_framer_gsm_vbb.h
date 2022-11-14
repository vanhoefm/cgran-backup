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
#ifndef INCLUDED_PRBLK_FRAMER_GSM_VBB_H
#define INCLUDED_PRBLK_FRAMER_GSM_VBB_H

#include <pr_framer_vbb.h>
#include <vector>
#include "field_coder.h"
#include <boost/scoped_array.hpp>

class pr_framer_gsm_vbb;

typedef boost::shared_ptr<pr_framer_gsm_vbb> pr_framer_gsm_vbb_sptr;

/**
 * \brief Public constructor.
 *
 * \param frame_size The frame size.
 * \param pktseq_code The code used for the packet sequence numbers.
 * \param sync_code The syncronization sequence in the syncronization frames.
 * \param data_code The syncronization sequence in the normal frames.
 *
 * Requirements on the arguments are:
 *  - \p pktseq_code is an even-length code
 *  - field_coder::codeword_length(pktseq_code) + \p sync_code.size() == \p frame_size
 */
pr_framer_gsm_vbb_sptr pr_make_framer_gsm_vbb(int frame_size, field_code_type pktseq_code,
			const std::vector<char>& sync_code, const std::vector<char>& data_code);

/**
 * \brief Framer for GSM-like frames
 *
 * \ingroup sigblk
 * The produced frames consist of a synchronization frame followed by a 
 * number of data frames. The synchronization frame is inserted when the
 * frame_seq field of the input meta data is zero. For more information,
 * see the design document.
 *
 * Ports
 *  - Input 0: (<b>\ref txmeta</b>, <b>char</b>[data_size])
 *  - Output 0: (<b>\ref txmeta</b>, <b>char</b>[frame_size])
 *
 * Above, \p data_size = \p frame_size - \p data_code.length().
 */
class pr_framer_gsm_vbb : public pr_framer_vbb
{
private:
	friend pr_framer_gsm_vbb_sptr pr_make_framer_gsm_vbb(int frame_size, field_code_type pktseq_code,
			const std::vector<char>& sync_code, const std::vector<char>& data_code);

	pr_framer_gsm_vbb(int frame_size, field_code_type pktseq_code,
			const std::vector<char>& sync_code, const std::vector<char>& data_code);

public:
	/**
	 * \brief Public destructor
	 */
	virtual ~pr_framer_gsm_vbb();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

private:
	field_code_type d_pktseq_code;
	std::vector<char> d_sync_code;
	std::vector<char> d_data_code;
	bool d_insertsync;
	boost::scoped_array<char> d_pktseq_cw;
};

#endif

