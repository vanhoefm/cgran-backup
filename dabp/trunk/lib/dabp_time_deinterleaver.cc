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

#include "dabp_time_deinterleaver.h"
#include <gr_io_signature.h>
#include <cstring>
#include <gruel/thread.h>

const int dabp_time_deinterleaver::DELAY_LEN[16]={16,8,12,4,14,6,10,2,15,7,11,3,13,5,9,1};

dabp_time_deinterleaver_sptr dabp_make_time_deinterleaver(int subchsz)
{
    return dabp_time_deinterleaver_sptr(new dabp_time_deinterleaver(subchsz));
}

dabp_time_deinterleaver::dabp_time_deinterleaver(int subchsz) 
    : gr_block("time_deinterleaver",
        gr_make_io_signature2(2,2,sizeof(float),sizeof(char)),
        gr_make_io_signature2(2,2,sizeof(float),sizeof(char))), d_M(subchsz*64)
{
    buf=new float*[d_M];
    for(int i=0;i<d_M;i++) {
        buf[i]=new float[DELAY_LEN[i%16]];
    }
    memset(idxbuf,0,sizeof(int)*16);
    set_output_multiple(64); // one CU at a time
    d_ibuf=new float[d_M];
}

dabp_time_deinterleaver::~dabp_time_deinterleaver()
{
    for(int i=0;i<d_M;i++)
        delete [] buf[i];
    delete [] buf;
    delete [] d_ibuf;
}

void dabp_time_deinterleaver::reset(int subchsz)
{
    gruel::scoped_lock guard(d_mutex); // avoid simultaneous access from work()
    // free old memories first
    for(int i=0;i<d_M;i++)
        delete [] buf[i];
    delete [] buf;
    delete [] d_ibuf;
    // set up for the new parameter
    d_M = subchsz*64;
    buf=new float*[d_M];
    for(int i=0;i<d_M;i++) {
        buf[i]=new float[DELAY_LEN[i%16]];
    }
    memset(idxbuf,0,sizeof(int)*16);
    d_ibuf = new float[d_M];
    d_icnt = d_ocnt = 0;
}

void dabp_time_deinterleaver::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%64==0);
    gruel::scoped_lock guard(d_mutex);
    int input_required=noutput_items; // almost sync_block
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_time_deinterleaver::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    int i,j;
    const float *in0=(const float*)input_items[0];
    const char *in1=(const char*)input_items[1];
    float *out0=(float*)output_items[0];
    char *out1=(char*)output_items[1];
    
    int nconsumed=0, nproduced=0;
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input
        if(d_icnt<d_M) { // ibuf not full, fill it
            if(d_icnt==0) { // check indicator
                while(nconsumed<ninput_items[0] && *in1 == 0) { // skip to start of subchannel
                    in0 ++;
                    in1 ++;
                    nconsumed++;
                }
                if(nconsumed==ninput_items[0]) { // nothing available from the input
                    break;
                }
            }
            if(d_icnt+ninput_items[0]-nconsumed<d_M) { // not enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ninput_items[0]-nconsumed)*sizeof(float));
                d_icnt += ninput_items[0]-nconsumed;
                nconsumed = ninput_items[0];
                break;
            }else {
                memcpy(d_ibuf+d_icnt, in0, (d_M-d_icnt)*sizeof(float));
                in0 += d_M-d_icnt;
                in1 += d_M-d_icnt;
                nconsumed += d_M-d_icnt;
                d_icnt = d_M;
                
                for(j=0;j<d_M;j++) { // store 
                    buf[j][idxbuf[j%16]]=d_ibuf[j];
                }
                for(j=0;j<16;j++) { // move indices
                    idxbuf[j]=(idxbuf[j]+1)%DELAY_LEN[j];
                }
            }
        }
        
        assert(d_icnt==d_M);
        if(d_M-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            for(j=0;j<noutput_items-nproduced;j++) {
                out0[j] = buf[d_ocnt+j][idxbuf[(d_ocnt+j)%16]];
            }
            memset(out1, 0, noutput_items-nproduced);
            if(d_ocnt==0)
                out1[0]=1;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            for(j=0;j<d_M-d_ocnt;j++) {
                out0[j] = buf[d_ocnt+j][idxbuf[(d_ocnt+j)%16]];
            }
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

