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
#ifndef INCLUDED_PR_SNR_ESTIMATE_C_H
#define INCLUDED_PR_SNR_ESTIMATE_C_H

#include <gr_sync_block.h>
#include <vector>
#include <boost/thread/mutex.hpp>
#include "modulation.h"

class pr_snr_estimate_c;

typedef boost::shared_ptr<pr_snr_estimate_c> pr_snr_estimate_c_sptr;

/**
 * \brief Public constructor.
 */
pr_snr_estimate_c_sptr pr_make_snr_estimate_c(modulation_type modulation, int block_size = 128);

/**
 * \brief 
 *
 * \ingroup sigblk
 *
 * Ports
 *  - Input 0:
 *  - Output 0:
 */
class pr_snr_estimate_c : public gr_sync_block
{
private:
	friend pr_snr_estimate_c_sptr pr_make_snr_estimate_c(modulation_type modulation, int block_size);

	pr_snr_estimate_c(modulation_type modulation, int block_size);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~pr_snr_estimate_c();

	int block_size() const;
	std::vector<double> snr();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

private:
	modulation_type d_modulation;
	int d_block_size;
	std::vector<double> d_snr;
	boost::mutex d_snr_lock;
};

#endif

