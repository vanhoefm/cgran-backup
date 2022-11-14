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

#include "dabp_depuncturer.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>
#include <gruel/thread.h>

// Table 29 on p. 131
const char dabp_depuncturer::VPI1[32]={1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI2[32]={1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI3[32]={1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI4[32]={1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI5[32]={1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI6[32]={1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI7[32]={1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0};
const char dabp_depuncturer::VPI8[32]={1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};
const char dabp_depuncturer::VPI9[32]={1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};
const char dabp_depuncturer::VPI10[32]={1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};
const char dabp_depuncturer::VPI12[32]={1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0};
const char dabp_depuncturer::VPI13[32]={1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0};
const char dabp_depuncturer::VPI14[32]={1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0};
const char dabp_depuncturer::VPI23[32]={1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0};
const char dabp_depuncturer::VPI24[32]={1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1};
const char dabp_depuncturer::VT[24]={1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};
const int dabp_depuncturer::OPTION_PROTECT_FACTOR[8]={12,8,6,4,27,21,18,15}; //table 7&8

dabp_depuncturer_sptr dabp_make_depuncturer(int subchsz, int optprot)
{
    return dabp_depuncturer_sptr(new dabp_depuncturer(subchsz, optprot));
}

dabp_depuncturer::dabp_depuncturer(int subchsz, int optprot)
    : gr_block("depuncturer", 
        gr_make_io_signature2(2,2,sizeof(float),sizeof(char)), 
        gr_make_io_signature2(2,2,sizeof(float),sizeof(char)))
{
    init(subchsz, optprot);
    d_ibuf = new float[d_M];
    d_obuf = new float[d_depunct_len];
    d_icnt = d_ocnt = 0;
    //set_relative_rate(d_depunct_len/d_M);
    set_output_multiple(24);
}

void dabp_depuncturer::init(int subchsz, int optprot)
{
    assert(optprot>=0 && optprot<=7);
    assert(subchsz%OPTION_PROTECT_FACTOR[optprot]==0);
    d_n=subchsz/OPTION_PROTECT_FACTOR[optprot];
    d_M=subchsz*64;
    switch(optprot) {
        case 0: // option==000, protection level==00 (profile 1-A)
            d_L1=6*d_n-3;
            d_L2=3;
            d_PI1=VPI24;
            d_PI2=VPI23;
            break;
        case 1: // option==000, protection level==01 (profile 2-A)
            if(d_n>1) {
                d_L1=2*d_n-3;
                d_L2=4*d_n+3;
                d_PI1=VPI14;
                d_PI2=VPI13;
            }else {
                d_L1=5;
                d_L2=1;
                d_PI1=VPI13;
                d_PI2=VPI12;
            }
            break;
        case 2: // option==000, protection level==10 (profile 3-A)
            d_L1=6*d_n-3;
            d_L2=3;
            d_PI1=VPI8;
            d_PI2=VPI7;
            break;
        case 3: // option==000, protection level==11 (profile 4-A)
            d_L1=4*d_n-3;
            d_L2=2*d_n+3;
            d_PI1=VPI3;
            d_PI2=VPI2;
            break;
            
        case 4: // option==001, protection level==00 (profile 1-B)
            d_L1=24*d_n-3;
            d_L2=3;
            d_PI1=VPI10;
            d_PI2=VPI9;
            break;
        case 5: // option==001, protection level==01 (profile 2-B)
            d_L1=24*d_n-3;
            d_L2=3;
            d_PI1=VPI6;
            d_PI2=VPI5;
            break;
        case 6: // option==001, protection level==10 (profile 3-B)
            d_L1=24*d_n-3;
            d_L2=3;
            d_PI1=VPI4;
            d_PI2=VPI3;
            break;
        case 7: // option==001, protection level==11 (profile 4-B)
            d_L1=24*d_n-3;
            d_L2=3;
            d_PI1=VPI2;
            d_PI2=VPI1;
            break;
        default:
            std::cerr<<"Unsupported error protection profile!"<<std::endl;
            assert(false);
    }
    d_I=32*(d_L1+d_L2);
    d_depunct_len=4*d_I+24;
}

dabp_depuncturer::~dabp_depuncturer()
{
    delete [] d_ibuf;
    delete [] d_obuf;
}

void dabp_depuncturer::reset(int subchsz, int optprot)
{
    gruel::scoped_lock guard(d_mutex); // avoid simultaneous access from work()
    delete [] d_ibuf;
    delete [] d_obuf;
    init(subchsz, optprot);
    d_ibuf = new float[d_M];
    d_obuf = new float[d_depunct_len];
    d_icnt = d_ocnt = 0;
}

void dabp_depuncturer::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%24==0);
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    int input_required=noutput_items*d_M/d_depunct_len;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_depuncturer::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%24==0);
    int i,j,l;
    const float *in0=(const float*)input_items[0];
    const char *in1=(const char*)input_items[1];
    float *out0=(float*)output_items[0];
    char *out1=(char*)output_items[1];
    
    int nconsumed=0, nproduced=0;
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input from in to ibuf
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
                
                // depuncture ibuf->obuf
                int idxi=0, idxo=0;
                for(j=0;j<d_L1;j++) { // first L1 blocks
                    for(l=0;l<128;l++) {
                        if(d_PI1[l%32]) {
                            d_obuf[idxo++]=d_ibuf[idxi++];
                        }else {
                            d_obuf[idxo++]=0;
                        }
                    }
                }
                for(j=0;j<d_L2;j++) { // remaining L2 blocks
                    for(l=0;l<128;l++) {
                        if(d_PI2[l%32]) {
                            d_obuf[idxo++]=d_ibuf[idxi++];
                        }else {
                            d_obuf[idxo++]=0;
                        }
                    }
                }
                for(l=0;l<24;l++) { // tail bits
                    if(VT[l]) {
                        d_obuf[idxo++]=d_ibuf[idxi++];
                    }else {
                        d_obuf[idxo++]=0;
                    }
                }
                assert(idxi==d_M && idxo==d_depunct_len);
            }
        }
        
        // from obuf to out
        assert(d_icnt==d_M);
        if(d_depunct_len-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            memcpy(out0, d_obuf+d_ocnt, (noutput_items-nproduced)*sizeof(float));
            memset(out1, 0, noutput_items-nproduced);
            if(d_ocnt==0)
                out1[0]=1;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            memcpy(out0, d_obuf+d_ocnt, (d_depunct_len-d_ocnt)*sizeof(float));
            memset(out1, 0, d_depunct_len-d_ocnt);
            if(d_ocnt==0)
                out1[0]=1; // flag the first bit of a subchannel
            out0 += d_depunct_len-d_ocnt;
            out1 += d_depunct_len-d_ocnt;
            nproduced += d_depunct_len-d_ocnt;
            d_ocnt = d_icnt = 0; // clear the buffer d_i/obuf
        }
    }
    consume_each(nconsumed);
    return nproduced;
}

