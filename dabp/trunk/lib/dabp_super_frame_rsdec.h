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
#ifndef INCLUDED_DABP_SUPER_FRAME_RSDEC_H
#define INCLUDED_DABP_SUPER_FRAME_RSDEC_H
#include <gr_block.h>
#include "dabp_rscodec.h"
#include <boost/thread.hpp>

class dabp_super_frame_rsdec;

typedef boost::shared_ptr<dabp_super_frame_rsdec> dabp_super_frame_rsdec_sptr;

/*! Instantiate super frame RS decoder
 * \param subchidx subchannel index = logical frame length / 24
 */
dabp_super_frame_rsdec_sptr dabp_make_super_frame_rsdec(int subchidx);

/*! DAB+ super frame RS decoder
 * It takes 120*subchidx bytes and produces 110*subchidx at a time
 */
class dabp_super_frame_rsdec : public gr_block
{
    private:
    friend dabp_super_frame_rsdec_sptr dabp_make_super_frame_rsdec(int subchidx);
    dabp_super_frame_rsdec(int subchidx);
    int d_subchidx; // subchannel index
    dabp_rscodec d_rs;
    
    boost::mutex d_mutex;
    unsigned char * d_ibuf, * d_obuf;
    int d_icnt, d_ocnt;
    
    public:
    ~dabp_super_frame_rsdec();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
    void reset(int subchidx);
};

#endif // INCLUDED_DABP_SUPER_FRAME_RSDEC_H

