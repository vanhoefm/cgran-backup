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
#ifndef INCLUDED_PRBLK_DEFRAMER_GSM_VCC_H
#define INCLUDED_PRBLK_DEFRAMER_GSM_VCC_H

#include <pr_deframer_vcc.h>
#include <vector>
#include "field_code_type.h"
#include <boost/scoped_array.hpp>

class pr_deframer_gsm_vcc;

typedef boost::shared_ptr<pr_deframer_gsm_vcc> pr_deframer_gsm_vcc_sptr;

/**
 * Public constructor.
 *
 * \param frame_size Size of input.
 * \param pktseq_code ECC for packet sequence field.
 * \param sync_code Synchronization sequence in SYNC frames.
 * \param data_code Training sequence in DATA frames.
 */
pr_deframer_gsm_vcc_sptr pr_make_deframer_gsm_vcc(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code);

/**
 * \brief Deframer for GSM-like frames.
 *
 * \ingroup sigblk
 * The packet sequence field of SYNC frames is decoded, and the resulting
 * sequence number is stored. The SYNC frame is then dropped, and the stored
 * sequenc number is used in the \p pkt_seq field of the output headers for
 * the deframed DATA frames.
 *
 * Ports
 *  - Input 0: (<b>rxframe</b>, <b>complex</b>[frame_size])
 *  - Output 0: (<b>rxframe</b>, <b>complex</b>[data_size])
 *
 * \p data_size = \p frame_size - \p data_code.size()
 */
class pr_deframer_gsm_vcc : public pr_deframer_vcc
{
private:
	friend pr_deframer_gsm_vcc_sptr pr_make_deframer_gsm_vcc(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code);

	pr_deframer_gsm_vcc(int frame_size, field_code_type pktseq_code, const std::vector<char>& sync_code, const std::vector<char>& data_code);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_deframer_gsm_vcc();

protected:
	virtual bool deframe(const rxframe& inmeta, const gr_complex* in,
			rxframe& outmeta, gr_complex* out);

private:
	field_code_type d_pktseq_code;
	std::vector<char> d_sync_code;
	std::vector<char> d_data_code;
	bool d_syncvalid;
	unsigned int d_pktseq;
	boost::scoped_array<float> d_pktseq_cw;
};

#endif

