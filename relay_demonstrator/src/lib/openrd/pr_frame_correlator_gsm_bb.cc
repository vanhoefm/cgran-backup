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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "openrd_debug.h"

#include "pr_frame_correlator_gsm_bb.h"
#include "gsm_frame.h"
#include <gr_io_signature.h>

pr_frame_correlator_gsm_bb_sptr pr_make_frame_correlator_gsm_bb(int input_size, int frame_size, const std::vector<char>& sync_code, int sync_nrequired, const std::vector<char>& data_code, int data_nrequired)
{
	return pr_frame_correlator_gsm_bb_sptr(new pr_frame_correlator_gsm_bb(input_size, frame_size, sync_code, sync_nrequired, data_code, data_nrequired));
}

static const int MIN_IN = 1;
static const int MAX_IN = 1;
static const int MIN_OUT = 1;
static const int MAX_OUT = 1;

pr_frame_correlator_gsm_bb::pr_frame_correlator_gsm_bb(int input_size, int frame_size, const std::vector<char>& sync_code, int sync_nrequired, const std::vector<char>& data_code, int data_nrequired) :
	pr_frame_correlator_bb(input_size, frame_size),
	d_sync_code(sync_code), d_sync_nrequired(sync_nrequired),
	d_data_code(data_code), d_data_nrequired(data_nrequired),
	d_syncfound(false), d_cnt(0), d_framecnt(0), d_data_seek_radius(4), 
	d_data_nmissed(0), d_data_maxmissed(3), d_maxdataframes(65535)
{
	if(frame_size % 2 != 0)
		throw "pr_frame_correlator_gsm_bb : frame_size must be even";

	if(sync_code.size() % 2 != 0)
		throw "pr_frame_correlator_gsm_bb : length of sync_code must be even";

	if(data_code.size() % 2 != 0)
		throw "pr_frame_correlator_gsm_bb : length of data_code must be even";

	d_sync_offset = (frame_size-sync_code.size())/2;
	d_data_offset = (frame_size-data_code.size())/2;
}

pr_frame_correlator_gsm_bb::~pr_frame_correlator_gsm_bb()
{
}

int pr_frame_correlator_gsm_bb::correlate(const char* data)
{
	int corr = 0;

	// Always look for sync frame
	for(unsigned int i = 0; i < d_sync_code.size(); i++)
		corr += (data[d_sync_offset+i] == d_sync_code[i]);

	// If sync frame is found, set bit counter and return
	if(corr >= d_sync_nrequired)
	{
		d_syncfound = true;
		d_cnt = 0;
		d_framecnt = 0;
		d_data_nmissed = 0;
		return GSM_FRAME_SYNC;
	}

	// If sync has not been achieved, don't bother looking for a data frame
	if(!d_syncfound)
		return 0;

	d_cnt += input_size();

	// If we are far before the expected position, don't bother looking
	if(d_cnt < (int)frame_size()-d_data_seek_radius)
		return 0;

	// If we are far behind the expected position, skip this frame
	if(d_cnt > (int)frame_size()+d_data_seek_radius)
	{
		d_cnt -= frame_size();
		d_framecnt++;
		if(d_framecnt == d_maxdataframes)
			d_syncfound = false;
		else
		{
			d_data_nmissed++;
			if(d_data_nmissed == d_data_maxmissed)
				d_syncfound = false;
		}
		return 0;
	}
	
	// Otherwise, look for a data frame
	corr = 0;
	for(unsigned int i = 0; i < d_data_code.size(); i++)
		corr += (data[d_data_offset+i] == d_data_code[i]);

	// If found, return frame type and frame counter
	if(corr >= d_data_nrequired)
	{
		int retval = GSM_FRAME_DATA | d_framecnt;

		d_cnt -= frame_size();
		d_framecnt++;
		if(d_framecnt == d_maxdataframes)
			d_syncfound = false;

		d_data_nmissed = 0;

		return retval;
	}

	return 0;
}

void pr_frame_correlator_gsm_bb::set_maxdataframes(int maxdataframes)
{
	d_maxdataframes = maxdataframes;
}

