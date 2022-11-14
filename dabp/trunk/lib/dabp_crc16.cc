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

#include "dabp_crc16.h"
#include <cstring>

// g(x)=1+x^5+x^12+x^16
const unsigned char dabp_crc16::g[16]={1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0};

dabp_crc16::dabp_crc16()
{
    // prepare the table
    unsigned char regs[16];
    int i,j;
    unsigned short itab[8];
    for(i=0;i<8;i++) {
        memset(regs,0,16);
        regs[8+i]=1;
        itab[i]=run8(regs);
    }
    for(i=0;i<256;i++) {
        tab[i]=0;
        for(j=0;j<8;j++) {
            if(i&(1<<j))
                tab[i]=tab[i]^itab[j];
        }
    }
}

dabp_crc16::~dabp_crc16()
{
}

unsigned short dabp_crc16::run8(unsigned char regs[])
{
    int i,j;
    unsigned char z;
    for(i=0;i<8;i++) {
        z=regs[15];
        for(j=15;j>0;j--)
            regs[j]=regs[j-1]^(z&g[j]);
        regs[0]=z;
    }
    unsigned short v=0;
    for(i=15;i>=0;i--)
        v=(v<<1)|regs[i];
    return v;
}

bool dabp_crc16::check(const unsigned char *x, int data_size)
{
    int i;
    unsigned short state=0xffff; // initialize to all 1s
    unsigned short istate;
    for(i=0;i<data_size;i++) { // for the data part
        istate=tab[(state>>8)^x[i]];
        state=istate^(state<<8);
    }
    for(;i<data_size+2;i++) { // for the two byte parity
        istate=tab[(state>>8)^((~x[i])&0xff)];
        state=istate^(state<<8);
    }
    return state==0;
}

void dabp_crc16::generate(unsigned char *x, int data_size)
{
    int i;
    unsigned short state=0xffff; // initialize to all 1s
    unsigned short istate;
    for(i=0;i<data_size;i++) { // for the data part
        istate=tab[(state>>8)^x[i]];
        state=istate^(state<<8);
    }
    x[i]=~(state>>8);
    x[i+1]=~(state&0xff);
}

