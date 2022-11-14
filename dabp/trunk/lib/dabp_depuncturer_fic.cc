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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dabp_depuncturer_fic.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>

const char dabp_depuncturer_fic::VPI16[32]={1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0};
const char dabp_depuncturer_fic::VPI15[32]={1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0};
const char dabp_depuncturer_fic::VT[24]={1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};

dabp_depuncturer_fic_sptr dabp_make_depuncturer_fic(int mode)
{
    return dabp_depuncturer_fic_sptr(new dabp_depuncturer_fic(mode));
}

dabp_depuncturer_fic::dabp_depuncturer_fic(int mode)
    : gr_block("depuncturer_fic", 
        gr_make_io_signature(1,1,sizeof(float)), 
        gr_make_io_signature(1,1,sizeof(float))), d_mode(mode)
{
    switch(mode) {
        case 1:
        case 2:
        case 4:
            d_I=768; // bits at the output of scrambler for 3 FIBs
            d_L1=21; // 21 blocks PI=16
            d_L2=3; // 3 blocks PI=15
            d_M=2304; // bits at the output of puncturer for 3 FIBs
            break;
        case 3:
            d_I=1024; // bits at the output of scrambler for 4 FIBs
            d_L1=29; // 29 blocks PI=16
            d_L2=3; // 3 blocks PI=15
            d_M=3072; // bits at the output of puncturer for 4 FIBs
            break;
        default:
            std::cerr<<"Unknown transmission mode!"<<std::endl;
            assert(false);
    }
    d_PI1=VPI16;
    d_PI2=VPI15;
    d_depunct_len=4*d_I+24; // bits at output of conv encoder but before puncturing
    set_relative_rate((float)d_depunct_len/d_M);
    set_output_multiple(d_depunct_len);
}

dabp_depuncturer_fic::~dabp_depuncturer_fic()
{
}

void dabp_depuncturer_fic::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%d_depunct_len==0);
    int input_required=(noutput_items/d_depunct_len)*d_M;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_depuncturer_fic::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%d_depunct_len==0);
    int nblocks=noutput_items/d_depunct_len;
    const float *in=(const float*)input_items[0];
    float *out=(float*)output_items[0];
    int i,j,l;
    
    for(i=0;i<nblocks;i++) {
        for(j=0;j<d_L1;j++) { // first L1 blocks
            for(l=0;l<128;l++) {
                if(d_PI1[l%32]) {
                    *out=*in;
                    in++;
                    out++;
                }else {
                    *out=0;
                    out++;
                }
            }
        }
        for(j=0;j<d_L2;j++) { // remaining L2 blocks
            for(l=0;l<128;l++) {
                if(d_PI2[l%32]) {
                    *out=*in;
                    in++;
                    out++;
                }else {
                    *out=0;
                    out++;
                }
            }
        }
        for(l=0;l<24;l++) { // tail bits
            if(VT[l]) {
                *out=*in;
                in++;
                out++;
            }else {
                *out=0;
                out++;
            }
        }
    }
    consume_each(nblocks*d_M);
    return noutput_items;
}

