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
#ifndef INCLUDED_DABP_SUPER_FRAME_SYNC_H
#define INCLUDED_DABP_SUPER_FRAME_SYNC_H
#include <gr_block.h>
#include "dabp_firecode.h"
#include <boost/thread.hpp>

class dabp_super_frame_sync;

typedef boost::shared_ptr<dabp_super_frame_sync> dabp_super_frame_sync_sptr;

/*! Instantiate DAB+ super frame synchronizer
 * \param len_logfrm length of a DAB logical frame in bytes (=I/8)
 */
dabp_super_frame_sync_sptr dabp_make_super_frame_sync(int len_logfrm);

/*! \brief DAB+ super frame synchronizer
 * It takes in one or multiple logical frames at a time
 * It produces none or multiple super frames at a time
 */
class dabp_super_frame_sync : public gr_block
{
    private:
    friend dabp_super_frame_sync_sptr dabp_make_super_frame_sync(int len_logfrm);
    dabp_super_frame_sync(int len_logfrm);
    int d_len_logfrm; // length of a logical frame (bytes) = I/8
    int d_subchidx; // subchannel index = len_logfrm/24
    int d_sync; // state of sync
    dabp_firecode d_firecode;
    static const int MAX_RELIABILITY;
    static const int LOGFRMS_PER_SUPFRM;
    boost::mutex d_mutex;
    unsigned char * d_ibuf;
    int d_icnt, d_ocnt, d_frmcnt;
    
    public:
    ~dabp_super_frame_sync();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
    int get_subchidx() const { return d_subchidx; }
    void reset(int len_logfrm);
};

#endif // INCLUDED_DABP_SUPER_FRAME_SYNC_H

