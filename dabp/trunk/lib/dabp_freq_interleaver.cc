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

#include "dabp_freq_interleaver.h"
#include <cstring>
#include <iostream>
#include <cassert>

dabp_freq_interleaver::dabp_freq_interleaver(int mode, int Nfft): d_Nfft(Nfft), d_F(NULL)
{
    int i,k;
    // prepare the table
    switch(mode) {
        case 1:
        d_K=1536;
        d_N=2048;
        d_F=new int[d_K];
        d_A=new int[d_N];
        d_A[0]=0;
        k=0;
        for(i=1;i<d_N;i++) {
            d_A[i]=(13*d_A[i-1]+511)%2048;
            if(d_A[i]>=256 && d_A[i]<=1792 && d_A[i]!=1024) {
                d_F[k]=d_A[i]-1024;
                k++;
            }
        }
        assert(k==d_K);
        delete [] d_A;
        break;
        
        case 2:
        d_K=384;
        d_N=512;
        d_F=new int[d_K];
        d_A=new int[d_N];
        d_A[0]=0;
        k=0;
        for(i=1;i<d_N;i++) {
            d_A[i]=(13*d_A[i-1]+127)%512;
            if(d_A[i]>=64 && d_A[i]<=448 && d_A[i]!=256) {
                d_F[k]=d_A[i]-256;
                k++;
            }
        }
        assert(k==d_K);
        delete [] d_A;
        break;
        
        case 3:
        d_K=192;
        d_N=256;
        d_F=new int[d_K];
        d_A=new int[d_N];
        d_A[0]=0;
        k=0;
        for(i=1;i<d_N;i++) {
            d_A[i]=(13*d_A[i-1]+63)%256;
            if(d_A[i]>=32 && d_A[i]<=224 && d_A[i]!=128) {
                d_F[k]=d_A[i]-128;
                k++;
            }
        }
        assert(k==d_K);
        delete [] d_A;
        break;
        
        case 4:
        d_K=768;
        d_N=1024;
        d_F=new int[d_K];
        d_A=new int[d_N];
        d_A[0]=0;
        k=0;
        for(i=1;i<d_N;i++) {
            d_A[i]=(13*d_A[i-1]+255)%1024;
            if(d_A[i]>=128 && d_A[i]<=896 && d_A[i]!=512) {
                d_F[k]=d_A[i]-512;
                k++;
            }
        }
        assert(k==d_K);
        delete [] d_A;
        break;
        
        default:
        std::cerr<<"Unkown transmission mode-"<<mode<<std::endl;
        assert(false);
    }
}

dabp_freq_interleaver::~dabp_freq_interleaver()
{
    delete [] d_F;
}

int dabp_freq_interleaver::interleave(int idx)
{
    return d_F[idx]>=0 ? d_F[idx] : d_F[idx]+d_Nfft;
}


