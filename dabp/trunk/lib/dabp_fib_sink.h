/* -*- c++ -*- */
/*
 * Copyright 2004,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_DABP_FIB_SINK_H
#define INCLUDED_DABP_FIB_SINK_H

#include <gr_sync_block.h>
#include "dabp_crc16.h"
#include "dabp_fib_constants.h"

struct subch_info {
	unsigned char subchid; // 0-63
	unsigned int sid; // service id
	unsigned char label[17];
	unsigned short start_addr;
	unsigned short subchsz;
	unsigned char option;
	unsigned char protect_lev;
};

class dabp_fib_sink;
typedef boost::shared_ptr<dabp_fib_sink> dabp_fib_sink_sptr;

dabp_fib_sink_sptr dabp_make_fib_sink();

/*!
 * \brief sink for DAB FIBs
 *
 * One input: FIB byte stream
 *
 * \ingroup sink
 */
class dabp_fib_sink : public gr_sync_block
{
	friend dabp_fib_sink_sptr dabp_make_fib_sink();

private:
	void dump_fib(const unsigned char *fib);
	int process_fib(const unsigned char *fib);
	int process_fig(unsigned char type, const unsigned char *data, unsigned char length);
	dabp_crc16 crc;
	struct subch_info d_subch[MAX_NUM_SUBCH];
	int update_label(unsigned int sid, unsigned char *label);
	void update_service_org(unsigned char subchid, unsigned int sid);
	void update_subch_org(unsigned char subchid, unsigned short start_addr, unsigned short subchsz, unsigned char option, unsigned char protect_lev);
	unsigned char d_ens_label[17];
	int d_duplicate[MAX_NUM_SUBCH];
	
protected:
	dabp_fib_sink();

public:
	~dabp_fib_sink();
	void print_subch();
    void save_subch(const char *filename);
	int work(int noutput_items,
			 gr_vector_const_void_star &input_items,
			 gr_vector_void_star &output_items);
};

#endif /* INCLUDED_DABP_FIB_SINK_H */
