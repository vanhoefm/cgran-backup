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

#include "dabp_super_frame_rsdec.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <gruel/thread.h>

dabp_super_frame_rsdec_sptr dabp_make_super_frame_rsdec(int subchidx)
{
    return dabp_super_frame_rsdec_sptr(new dabp_super_frame_rsdec(subchidx));
}

dabp_super_frame_rsdec::dabp_super_frame_rsdec(int subchidx)
    : gr_block("super_frame_rsdec", 
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char)), 
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char))), d_subchidx(subchidx)
{
    // output: subchidx*110 RS decoded bytes sequence
    set_relative_rate(110.0/120.0);
    set_output_multiple(110);
    d_ibuf = new unsigned char[d_subchidx*120];
    d_obuf = new unsigned char[d_subchidx*110];
    d_icnt = d_ocnt = 0;
}

dabp_super_frame_rsdec::~dabp_super_frame_rsdec()
{
    delete [] d_ibuf;
    delete [] d_obuf;
}

void dabp_super_frame_rsdec::reset(int subchidx)
{
    gruel::scoped_lock guard(d_mutex);
    delete [] d_ibuf;
    delete [] d_obuf;
    d_subchidx = subchidx;
    d_ibuf = new unsigned char[d_subchidx*120];
    d_obuf = new unsigned char[d_subchidx*110];
    d_icnt = d_ocnt = 0;
}

void dabp_super_frame_rsdec::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%110==0);
    int input_required=noutput_items/110*120;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_super_frame_rsdec::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%110==0);
    const unsigned char *in0=(const unsigned char*)input_items[0];
    unsigned char *out0=(unsigned char *)output_items[0];
    const char *in1=(const char*)input_items[1];
    char *out1=(char *)output_items[1];
    
    int nconsumed=0, nproduced=0; // the number of consumed input items & produced output items
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    const int ilen=d_subchidx*120, olen=d_subchidx*110;
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input from in to ibuf
        if(d_icnt<ilen) { // ibuf not full, fill it
            if(d_icnt==0) { // check indicator
                while(nconsumed<ninput_items[0] && *in1 == 0) { // skip to start of super frame
                    in0 ++;
                    in1 ++;
                    nconsumed++;
                }
                if(nconsumed==ninput_items[0]) { // nothing available from the input
                    break;
                }
            }
            if(d_icnt+ninput_items[0]-nconsumed<ilen) { // not enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ninput_items[0]-nconsumed)*sizeof(unsigned char));
                d_icnt += ninput_items[0]-nconsumed;
                nconsumed = ninput_items[0];
                break;
            }else { // enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ilen-d_icnt)*sizeof(unsigned char));
                in0 += ilen-d_icnt;
                in1 += ilen-d_icnt;
                nconsumed += ilen-d_icnt;
                d_icnt = ilen;
                
                // decode ibuf->obuf
                int i, j, k;
                unsigned char r[120]; // received codeword
                unsigned char d[110]; // decoded info bytes
                for(j=0;j<d_subchidx;j++) { // for each RS code word
                    for(k=0;k<120;k++) { // copy one received code word in
                        r[k]=d_ibuf[j+k*d_subchidx];
                    }
                    d_rs.dec(r,d);
                    for(k=0;k<110;k++) // copy the decoded info bytes out
                        d_obuf[j+k*d_subchidx]=d[k];
                }
            }
        }
        
        // from obuf to out
        assert(d_icnt==ilen);
        if(olen-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            memcpy(out0, d_obuf+d_ocnt, (noutput_items-nproduced)*sizeof(unsigned char));
            memset(out1, 0, noutput_items-nproduced);
            if(d_ocnt==0)
                out1[0]=1;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            memcpy(out0, d_obuf+d_ocnt, (olen-d_ocnt)*sizeof(unsigned char));
            memset(out1, 0, olen-d_ocnt);
            if(d_ocnt==0)
                out1[0]=1; // flag the first bit of a subchannel
            out0 += olen-d_ocnt;
            out1 += olen-d_ocnt;
            nproduced += olen-d_ocnt;
            d_ocnt = d_icnt = 0; // clear the buffer d_i/obuf
        }
    }
    consume_each(nconsumed);
    return nproduced;
}

