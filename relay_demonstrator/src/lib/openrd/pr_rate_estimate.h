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
#ifndef INCLUDED_PR_RATE_ESTIMATE_H
#define INCLUDED_PR_RATE_ESTIMATE_H

#include <gr_sync_block.h>
#include <boost/thread/mutex.hpp>

class pr_rate_estimate;

typedef boost::shared_ptr<pr_rate_estimate> pr_rate_estimate_sptr;

/**
 * \brief Public constructor.
 *
 * \param item_size Size of input vector
 * \param numavg Number of samples to average
 */
pr_rate_estimate_sptr pr_make_rate_estimate(size_t item_size);

/**
 * \brief Rate estimator.
 *
 * \ingroup sigblk
 * This block computes the average number of elements per second.
 *
 * Ports
 *  - Input 0: <b>char</b>[item_size]
 */
class pr_rate_estimate : public gr_sync_block
{
private:
	friend pr_rate_estimate_sptr pr_make_rate_estimate(size_t item_size);

	pr_rate_estimate(size_t item_size);

public:
	/**
	 * \brief Public destructor.
	 */
	virtual ~pr_rate_estimate();

	virtual int work(int noutput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

	/**
	 * \returns estimation of rate in items/sec
	 */
	double rate();

	/**
	 * Clears the rate estimate.
	 */
	void clear();

private:
	struct timeval d_stamp;
	int d_count;
	double d_rate;
	mutable boost::mutex d_count_lock;
};

#endif

