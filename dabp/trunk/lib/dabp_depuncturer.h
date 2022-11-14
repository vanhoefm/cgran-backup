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

#ifndef INCLUDED_DABP_DEPUNCTURER_H
#define INCLUDED_DABP_DEPUNCTURER_H
#include <gr_block.h>
#include <boost/thread.hpp>

class dabp_depuncturer;

typedef boost::shared_ptr<dabp_depuncturer> dabp_depuncturer_sptr;

/*! Instantiate DAB depuncturer
 * \param subchsz subchannel size
 * \param optprot = option<<2 | protection_level
 */
dabp_depuncturer_sptr dabp_make_depuncturer(int subchsz, int optprot);

/*! \brief DAB depuncturer for MSC
 * It takes in M=subchsz*64 softbits and produces 4*I+24 softbits
 */
class dabp_depuncturer : public gr_block
{
    private:
    friend dabp_depuncturer_sptr dabp_make_depuncturer(int subchsz, int optprot);
    dabp_depuncturer(int subchsz, int optprot);
    void init(int subchsz, int optprot);
    int d_n, d_L1, d_L2, d_I, d_depunct_len, d_M;
    const char *d_PI1, *d_PI2;
    
    static const char VPI1[32], VPI2[32], VPI3[32], VPI4[32], VPI5[32], VPI6[32], 
        VPI7[32], VPI8[32], VPI9[32], VPI10[32], VPI12[32], VPI13[32], VPI14[32], 
        VPI23[32], VPI24[32], VT[24];
    static const int OPTION_PROTECT_FACTOR[8];
    
    boost::mutex d_mutex;
    float * d_ibuf, * d_obuf;
    int d_icnt, d_ocnt;
    
    public:
    ~dabp_depuncturer();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
    int getI() const { return d_I;}
    void reset(int subchsz, int optprot);
};
#endif // INCLUDED_DABP_DEPUNCTURER_H

