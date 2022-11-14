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
#ifndef INCLUDED_DABP_SUBCHANNEL_SELECTOR_H
#define INCLUDED_DABP_SUBCHANNEL_SELECTOR_H
#include <gr_block.h>
#include <boost/thread.hpp>

class dabp_subchannel_selector;

typedef boost::shared_ptr<dabp_subchannel_selector> dabp_subchannel_selector_sptr;

/*! Instantiate subchannel selector
 * \param cifsz size of one CIF
 * \param start_addr subchannel start address
 * \param subchsz subchannel size
 */
dabp_subchannel_selector_sptr dabp_make_subchannel_selector(int cifsz,int start_addr,int subchsz);

/*! \brief DAB subchannel selector (MSC)
 * It takes in one CIF of cifsz soft bits and produces one subchannel of M=subchsz*64 soft bits at a time
 * On-the-fly changes to start_addr and subchsz are allowed
 * The second output is an indicator to the first bit of the M-bit subchannel
 */
class dabp_subchannel_selector : public gr_block
{
    private:
    friend dabp_subchannel_selector_sptr dabp_make_subchannel_selector(int cifsz, int start_addr, int subchsz);
    dabp_subchannel_selector(int cifsz, int start_addr, int subchsz);
    int d_cifsz, d_start_addr, d_subchsz;
    int d_M;
    int d_icnt, d_ocnt;
    float *d_buf;
    boost::mutex d_mutex;
    
    public:
    ~dabp_subchannel_selector();
    void reset(int start_addr, int subchsz);
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};
#endif // INCLUDED_DABP_SUBCHANNEL_SELECTOR_H

