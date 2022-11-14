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

#include "dabp_super_frame_sync.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <gruel/thread.h>

const int dabp_super_frame_sync::MAX_RELIABILITY=4;
const int dabp_super_frame_sync::LOGFRMS_PER_SUPFRM=5;

dabp_super_frame_sync_sptr dabp_make_super_frame_sync(int len_logfrm)
{
    return dabp_super_frame_sync_sptr(new dabp_super_frame_sync(len_logfrm));
}

dabp_super_frame_sync::dabp_super_frame_sync(int len_logfrm)
    : gr_block("super_frame_sync", 
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char)), 
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char))), 
    d_len_logfrm(len_logfrm), d_subchidx(len_logfrm/24), d_sync(-1)
{
    assert(d_len_logfrm%24==0); // this must be true for both EEP A and B
    // output: subchidx*120 RS coded byte sequence comprising a super frame
    set_relative_rate(1);
    set_output_multiple(120);
    d_ibuf = new unsigned char[d_len_logfrm];
    d_icnt = d_ocnt = d_frmcnt = 0;
}

dabp_super_frame_sync::~dabp_super_frame_sync()
{
    delete [] d_ibuf;
}

void dabp_super_frame_sync::reset(int len_logfrm)
{
    assert(len_logfrm%24==0);
    gruel::scoped_lock guard(d_mutex); // avoid simultaneous access from work()
    delete [] d_ibuf;
    d_len_logfrm=len_logfrm;
    d_ibuf = new unsigned char[d_len_logfrm];
    d_sync = -1;
    d_icnt = d_ocnt = d_frmcnt = 0;
}

void dabp_super_frame_sync::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%120==0);
    int input_required=noutput_items;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_super_frame_sync::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%120==0);
    const unsigned char *in0=(const unsigned char*)input_items[0];
    unsigned char *out0=(unsigned char *)output_items[0];
    const char *in1=(const char*)input_items[1];
    char *out1=(char *)output_items[1];
    
    int nconsumed=0, nproduced=0; // the number of consumed input items & produced output items
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input from in to ibuf
        if(d_icnt<d_len_logfrm) { // ibuf not full, fill it
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
            if(d_icnt+ninput_items[0]-nconsumed<d_len_logfrm) { // not enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ninput_items[0]-nconsumed)*sizeof(unsigned char));
                d_icnt += ninput_items[0]-nconsumed;
                nconsumed = ninput_items[0];
                break;
            }else { // enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (d_len_logfrm-d_icnt)*sizeof(unsigned char));
                in0 += d_len_logfrm-d_icnt;
                in1 += d_len_logfrm-d_icnt;
                nconsumed += d_len_logfrm-d_icnt;
                d_icnt = d_len_logfrm;
                
                // sync ibuf->obuf
                if(d_sync<0) { // not synchronized yet
                    if(!d_firecode.check(d_ibuf)) { // not the 1st frame
                        d_icnt = 0;
                        continue;
                    }
                    
                    // 1st frame found
                    std::cerr<<"Super frame synchronization achieved!"<<std::endl;
                    d_sync=MAX_RELIABILITY;
                    d_frmcnt=0;
                    continue;
                }
                
                // already in sync
                if(d_frmcnt==0) {
                    if(!d_firecode.check(d_ibuf)) {
                        d_sync--;
                    }else{
                        d_sync=MAX_RELIABILITY;
                    }
                    if(d_sync<0) { // go out of sync
                        std::cerr<<"Super frame gets out of Sync!"<<std::endl;
                        d_icnt = 0;
                        continue;
                    }
                }
            }
        }
        
        // from obuf to out
        assert(d_icnt==d_len_logfrm);
        if(d_len_logfrm-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            memcpy(out0, d_ibuf+d_ocnt, (noutput_items-nproduced)*sizeof(unsigned char));
            memset(out1, 0, noutput_items-nproduced);
            if(d_frmcnt==0 && d_ocnt==0)
                out1[0]=1;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            memcpy(out0, d_ibuf+d_ocnt, (d_len_logfrm-d_ocnt)*sizeof(unsigned char));
            memset(out1, 0, d_len_logfrm-d_ocnt);
            if(d_frmcnt==0 && d_ocnt==0)
                out1[0]=1; // flag the first bit of a subchannel
            out0 += d_len_logfrm-d_ocnt;
            out1 += d_len_logfrm-d_ocnt;
            nproduced += d_len_logfrm-d_ocnt;
            d_ocnt = d_icnt = 0; // clear the buffer d_i/obuf
            d_frmcnt = (d_frmcnt+1)%LOGFRMS_PER_SUPFRM;
        }
    }
    consume_each(nconsumed);
    return nproduced;
}
