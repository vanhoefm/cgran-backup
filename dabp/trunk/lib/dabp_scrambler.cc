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

#include "dabp_scrambler.h"
#include <gr_io_signature.h>
#include <cassert>
#include <cstring>
#include <gruel/thread.h>

dabp_scrambler_sptr dabp_make_scrambler(int I)
{
    return dabp_scrambler_sptr(new dabp_scrambler(I));
}

dabp_scrambler::dabp_scrambler(int I) 
    : gr_block("scrambler",
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char)),
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char))), d_nbytes(I/8)
{
    assert(I%8==0);
    init_tab();
    init_prbs();
    d_ibuf = new unsigned char[d_nbytes];
    d_icnt = d_ocnt = 0;
    set_output_multiple(8);
}

void dabp_scrambler::reset(int I)
{
    assert(I%8==0);
    gruel::scoped_lock guard(d_mutex); // avoid simultaneous access from work()
    delete [] d_ibuf;
    delete [] prbs;
    d_nbytes = I/8;
    init_prbs();
    d_ibuf = new unsigned char[d_nbytes];
    d_icnt = d_ocnt = 0;
}

void dabp_scrambler::init_tab()
{
    // prepare the table
    unsigned char regs[9];
    int i,j;
    unsigned short itab[9];
    for(i=0;i<9;i++) {
        memset(regs,0,9);
        regs[i]=1;
        itab[i]=run8(regs);
    }
    for(i=0;i<512;i++) {
        tab[i]=0;
        for(j=0;j<9;j++) {
            if(i&(1<<j))
                tab[i]=tab[i]^itab[j];
        }
    }
}

void dabp_scrambler::init_prbs()
{
    prbs=new unsigned char[d_nbytes];
    // calculate the PRBS
    unsigned short state=0x1ff; // initialize the register to all 1s
    for(i=0;i<d_nbytes;i++) {
        prbs[i]=tab[state]&0x00ff;
        state=tab[state];
    }
}

unsigned short dabp_scrambler::run8(unsigned char regs[])
{
    int i,j;
    unsigned char z;
    for(i=0;i<8;i++) {
        z=regs[8]^regs[4];
        for(j=8;j>0;j--)
            regs[j]=regs[j-1];
        regs[0]=z;
    }
    unsigned short v=0;
    for(i=8;i>=0;i--)
        v=(v<<1)|regs[i];
    return v;
}

dabp_scrambler::~dabp_scrambler()
{
    delete [] d_ibuf;
    delete [] prbs;
}

void dabp_scrambler::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%8==0);
    int input_required=noutput_items;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_scrambler::general_work(int noutput_items,
                                   gr_vector_int &ninput_items,
                                   gr_vector_const_void_star &input_items,
                                   gr_vector_void_star &output_items)
{
    assert(noutput_items%8==0);
    int i,j,l;
    const unsigned char *in0=(const unsigned char*)input_items[0];
    const char *in1=(const char*)input_items[1];
    unsigned char *out0=(unsigned char*)output_items[0];
    char *out1=(char*)output_items[1];
    
    int nconsumed=0, nproduced=0;
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input from in to ibuf
        if(d_icnt<d_nbytes) { // ibuf not full, fill it
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
            if(d_icnt+ninput_items[0]-nconsumed<d_nbytes) { // not enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ninput_items[0]-nconsumed)*sizeof(unsigned char));
                d_icnt += ninput_items[0]-nconsumed;
                nconsumed = ninput_items[0];
                break;
            }else {
                memcpy(d_ibuf+d_icnt, in0, (d_nbytes-d_icnt)*sizeof(unsigned char));
                in0 += d_nbytes-d_icnt;
                in1 += d_nbytes-d_icnt;
                nconsumed += d_nbytes-d_icnt;
                d_icnt = d_nbytes;
            }
        }
        
        // from ibuf to out
        assert(d_icnt==d_nbytes);
        if(d_nbytes-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            for(i=0;i<noutput_items-nproduced;i++)
                out0[i]=d_ibuf[d_ocnt+i]^prbs[d_ocnt+i];
            memset(out1, 0, noutput_items-nproduced);
            if(d_ocnt==0)
                out1[0]=1;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            for(i=0;i<d_nbytes-d_ocnt;i++)
                out0[i]=d_ibuf[d_ocnt+i]^prbs[d_ocnt+i];
            memset(out1, 0, d_nbytes-d_ocnt);
            if(d_ocnt==0)
                out1[0]=1; // flag the first bit of a subchannel
            out0 += d_nbytes-d_ocnt;
            out1 += d_nbytes-d_ocnt;
            nproduced += d_nbytes-d_ocnt;
            d_ocnt = d_icnt = 0; // clear the buffer d_i/obuf
        }
    }
    consume_each(nconsumed);
    return nproduced;
}
