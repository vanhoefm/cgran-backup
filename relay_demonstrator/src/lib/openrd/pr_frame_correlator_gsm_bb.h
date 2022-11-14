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
#ifndef INCLUDED_PRBLK_FRAME_CORRELATOR_GSM_BB_H
#define INCLUDED_PRBLK_FRAME_CORRELATOR_GSM_BB_H

#include "pr_frame_correlator_bb.h"
#include <vector>

class pr_frame_correlator_gsm_bb;

typedef boost::shared_ptr<pr_frame_correlator_gsm_bb> pr_frame_correlator_gsm_bb_sptr;

/**
 * Public constructor.
 *
 * \param input_size Number of bits per symbol.
 * \param frame_size Number of bits per frame.
 * \param sync_code Synchronization sequence used for SYNC frames.
 * \param sync_nrequired Number of correct bits required for SYNC frames.
 * \param data_code Training sequence used for DATA frames.
 * \param data_nrequired Number of correct bits required for DATA frames.
 */
pr_frame_correlator_gsm_bb_sptr pr_make_frame_correlator_gsm_bb(int input_size, int frame_size, const std::vector<char>& sync_code, int sync_nrequired, const std::vector<char>& data_code, int data_nrequired);

/**
 * \brief Frame correlator for GSM-like frames.
 *
 * \ingroup sigblk
 * Every input is correlated to the \p sync_code. When \p sync_nrequired bits
 * are correct, GSM_FRAME_SYNC is output. As the SYNC frames are directly 
 * followed by DATA frames, correlation against \p data_code is only done in
 * a few symbols around the expected positions of the data frames. When a DATA
 * frame is found, the output contains the GSM_FRAME_SYNC flag and the frame
 * sequence number. The correlator tolerates three missed data frames until 
 * synchronization is dropped and correlation against DATA frames is aborted.
 *
 * The function set_maxdataframes() can be called to set the number of DATA
 * frames to look for after each SYNC frame.
 *
 * Ports
 *  - Input 0: <b>char</b>[input_size]
 *  - Output 0: <b>char</b>
 */
class pr_frame_correlator_gsm_bb : public pr_frame_correlator_bb
{
private:
	friend pr_frame_correlator_gsm_bb_sptr pr_make_frame_correlator_gsm_bb(int input_size, int frame_size, const std::vector<char>& sync_code, int sync_nrequired, const std::vector<char>& data_code, int data_nrequired);

	pr_frame_correlator_gsm_bb(int input_size, int frame_size, const std::vector<char>& sync_code, int sync_nrequired, const std::vector<char>& data_code, int data_nrequired);

public:
	/**
	 * Public destructor.
	 */
	virtual ~pr_frame_correlator_gsm_bb();

	/**
	 * Sets the number of DATA frames following each SYNC frame.
	 *
	 * \param maxdataframes Number of DATA frames.
	 */
	void set_maxdataframes(int maxdataframes);

protected:
	virtual int correlate(const char* data);

private:
	std::vector<char> d_sync_code;
	int d_sync_nrequired;
	std::vector<char> d_data_code;
	int d_data_nrequired;

	bool d_syncfound;
	int d_cnt;
	int d_framecnt;
	int d_data_seek_radius;
	int d_data_nmissed;
	int d_data_maxmissed;
	int d_sync_offset;
	int d_data_offset;
	int d_maxdataframes;
};

#endif

