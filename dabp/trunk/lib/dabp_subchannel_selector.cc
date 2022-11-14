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

#include "dabp_subchannel_selector.h"
#include <gr_io_signature.h>
#include <cassert>
#include <cstring>
#include <gruel/thread.h>

dabp_subchannel_selector_sptr dabp_make_subchannel_selector(int cifsz,int start_addr,int subchsz)
{
    return dabp_subchannel_selector_sptr(new dabp_subchannel_selector(cifsz,start_addr,subchsz));
}

dabp_subchannel_selector::dabp_subchannel_selector(int cifsz,int start_addr,int subchsz)
    : gr_block("subchannel_selector", 
        gr_make_io_signature(1,1,sizeof(float)), 
        gr_make_io_signature2(2,2,sizeof(float),sizeof(char))), 
    d_cifsz(cifsz), d_start_addr(start_addr), d_subchsz(subchsz), 
    d_M(subchsz*64), d_icnt(0), d_ocnt(0)
{
    d_buf = new float [d_cifsz];
    set_output_multiple(64); // one CU
}

dabp_subchannel_selector::~dabp_subchannel_selector()
{
    delete [] d_buf;
}

void dabp_subchannel_selector::reset(int start_addr, int subchsz)
{
    // ? thread safe ?
    gruel::scoped_lock guard(d_mutex); // avoid simultaneous access from work()
    d_start_addr = start_addr;
    d_subchsz = subchsz;
    d_M = d_subchsz*64;
    d_ocnt = 0;
}

void dabp_subchannel_selector::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%64==0);
    gruel::scoped_lock guard(d_mutex);
    int input_required=((noutput_items/64)*(d_cifsz/64)+d_subchsz-1)/d_subchsz*64;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_subchannel_selector::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    const float *in=(const float*)input_items[0];
    float *out0=(float*)output_items[0];
    char *out1=(char*)output_items[1];
    
    int nconsumed=0, nproduced=0;
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input
        if(d_icnt<d_cifsz) { // buffer not full, fill it
            if(d_icnt+ninput_items[0]-nconsumed<d_cifsz) { // not enough to fill the buffer
                memcpy(d_buf+d_icnt, in, (ninput_items[0]-nconsumed)*sizeof(float));
                //in += (ninput_items[0]-nconsumed);
                d_icnt += (ninput_items[0]-nconsumed);
                nconsumed = ninput_items[0];
                break;
            }else { // enough to fill the buffer
                memcpy(d_buf+d_icnt, in, (d_cifsz-d_icnt)*sizeof(float));
                in += d_cifsz-d_icnt;
                nconsumed = d_cifsz-d_icnt;
                d_icnt = d_cifsz;
            }
        }
        
        // process output
        // if program reaches here, the buffer must be full
        assert(d_icnt==d_cifsz);
        if(d_M-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            memcpy(out0, d_buf+d_start_addr*64+d_ocnt, (noutput_items-nproduced)*sizeof(float));
            memset(out1, 0, noutput_items-nproduced);
            if(d_ocnt==0)
                out1[0]=1; // flag the first bit of a subchannel
            //out0 += noutput_items-nproduced;
            //out1 += noutput_items-nproduced;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            memcpy(out0, d_buf+d_start_addr*64+d_ocnt, (d_M-d_ocnt)*sizeof(float));
            memset(out1, 0, d_M-d_ocnt);
            if(d_ocnt==0)
                out1[0]=1; // flag the first bit of a subchannel
            out0 += d_M-d_ocnt;
            out1 += d_M-d_ocnt;
            nproduced += d_M-d_ocnt;
            d_ocnt = d_icnt = 0; // clear the buffer d_buf
        }
    }
    consume_each(nconsumed);
    return nproduced;
}

