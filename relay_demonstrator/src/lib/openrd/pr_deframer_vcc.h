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
#ifndef INCLUDED_PRBLK_DEFRAMER_VCC_H
#define INCLUDED_PRBLK_DEFRAMER_VCC_H

#include <gr_block.h>

#include "stream_meta.h"
#include <vector>
#include <boost/thread/mutex.hpp>

class pr_deframer_vcc;

typedef boost::shared_ptr<pr_deframer_vcc> pr_deframer_vcc_sptr;

/**
 * \brief Base class for deframers which extracts data symbols from frame
 * encapsulations.
 *
 * \ingroup sigblk
 * The relevant meta information in the frames are put in the frame headers.
 *
 * Ports
 *  - Input 0: (<b>rxframe</b>, <b>complex</b>[frame_size])
 *  - Output 0: (<b>rxframe</b>, <b>complex</b>[data_size])
 */
class pr_deframer_vcc : public gr_block
{
protected:
	/**
	 * Protected constructor.
	 *
	 * \param frame_size Size of input.
	 * \param data_size Size of output.
	 */
	pr_deframer_vcc(int frame_size, int data_size);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_deframer_vcc();

	/**
	 * Returns the input size.
	 */
	int frame_size() const;

	/**
	 * Returns the output size.
	 */
	int data_size() const;

	/**
	 * Sets whether to store SNR estimation calculations in a vector that
	 * can be retrieved using snr(). The default is false.
	 *
	 * \param storesnr Store SNR estimation calculations.
	 */
	void set_storesnr(bool storesnr);

	/**
	 * Returns a vector of SNR measurements.
	 */
	std::vector<double> snr();

	virtual int general_work(int noutput_items,
			gr_vector_int& ninput_items,
			gr_vector_const_void_star& input_items,
			gr_vector_void_star& output_items);

protected:
	/**
	 * Extracts data symbols from a frame.
	 *
	 * \param inmeta Header of input frame.
	 * \param in Data of input frame.
	 * \param outmeta Header of output frame.
	 * \param out Data of output frame.
	 * \returns true if an output should be produced.
	 */
	virtual bool deframe(const rxframe& inmeta, const gr_complex* in,
			rxframe& outmeta, gr_complex* out) = 0;

	/**
	 * Sets whether SNR estimations should be calculated automatically by
	 * the base class. If set to true, the \p power field in the output
	 * header will be overwritten after the call to deframe(). The default
	 * is true.
	 *
	 * \param autosnr Automatic SNR estimation calculation.
	 */
	void set_autosnr(bool autosnr);

private:
	int d_frame_size;
	int d_data_size;
	bool d_storesnr;
	bool d_autosnr;
	std::vector<double> d_snr;
	boost::mutex d_snr_lock; // Protects d_snr
};

#endif

