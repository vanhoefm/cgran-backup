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
#ifndef INCLUDED_PR_BER_ESTIMATE_B_H
#define INCLUDED_PR_BER_ESTIMATE_B_H

#include <gr_sync_block.h>
#include <boost/thread/mutex.hpp>

class pr_ber_estimate_b;

typedef boost::shared_ptr<pr_ber_estimate_b> pr_ber_estimate_b_sptr;

/**
 * \brief Public constructor.
 *
 * \param alpha Parameter of averaging IIR filter.
 */
pr_ber_estimate_b_sptr pr_make_ber_estimate_b(double alpha = 0.0001);

/**
 * \brief Bit error rate estimation block.
 *
 * \ingroup sigblk
 * This block computes the bit error rate from two bit streams. The estimates
 * are passed through an averaging filter. The estimate can be accessed
 * through the ber() function, and cleared by the the clear() function.
 *
 * Ports
 *  - Input 0: <b>char</b>
 *  - Input 1: <b>char</b>
 */
class pr_ber_estimate_b : public gr_sync_block
{
private:
	friend pr_ber_estimate_b_sptr pr_make_ber_estimate_b(double alpha);

	pr_ber_estimate_b(double alpha);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~pr_ber_estimate_b();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * Updates the parameter of the averaging filter.
	 *
	 * \param alpha filter parameter
	 */
	void set_alpha(double alpha);

	/**
	 * \returns the current bit error rate estimate
	 */
	double ber() const;

	/**
	 * Clears the bit error rate to zero.
	 */
	void clear();

private:
	double d_alpha;
	double d_beta;
	double d_ber;
	mutable boost::mutex d_ber_lock; // Protects d_ber.
};

#endif

