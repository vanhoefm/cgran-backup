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
#ifndef INCLUDED_DABP_TIME_DEINTERLEAVER_H
#define INCLUDED_DABP_TIME_DEINTERLEAVER_H
#include <gr_block.h>
#include <boost/thread.hpp>

class dabp_time_deinterleaver;

typedef boost::shared_ptr<dabp_time_deinterleaver> dabp_time_deinterleaver_sptr;

/*! Instantiate time deinterleaver
 * \param subchsz subchannel size
 */
dabp_time_deinterleaver_sptr dabp_make_time_deinterleaver(int subchsz);

/*! \brief DAB time deinterleaver
 * It takes in M=subchsz*64 softbits at a time and is almost a sync_block
 * On-the-fly changes to subchsz are allowed
 */
class dabp_time_deinterleaver : public gr_block
{
    private:
    friend dabp_time_deinterleaver_sptr dabp_make_time_deinterleaver(int subchsz);
    dabp_time_deinterleaver(int subchsz);
    int d_M; // number of bits for subchannel size=64*subchsz
    float **buf; // point to arrays of delay lines' buffers
    int idxbuf[16]; // indices to buf
    static const int DELAY_LEN[16]; // length of delay lines
    float *d_ibuf; // input buffer
    int d_icnt, d_ocnt;
    boost::mutex d_mutex;
    
    public:
    ~dabp_time_deinterleaver();
    void reset(int subchsz);
    void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};

#endif // INCLUDED_DABP_TIME_DEINTERLEAVER_H

