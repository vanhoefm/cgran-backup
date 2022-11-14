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

#include "dabp_vitdec.h"
#include <gr_io_signature.h>
#include <iostream>
#include <cassert>
#include <gruel/thread.h>

const int dabp_vitdec::GENERATOR[4]={0133,0171,0145,0133};

dabp_vitdec_sptr dabp_make_vitdec(int I)
{
    return dabp_vitdec_sptr(new dabp_vitdec(I));
}

dabp_vitdec::dabp_vitdec(int I)
    : gr_block("vitdec", 
        gr_make_io_signature2(2,2,sizeof(float),sizeof(char)), 
        gr_make_io_signature2(2,2,sizeof(unsigned char),sizeof(char))), 
      d_I(I), d_fsm(1,4,std::vector<int>(GENERATOR,GENERATOR+4))
{
    assert(d_I%8==0);
    d_nbytes = d_I/8;
    //set_relative_rate(d_nbytes/(4.0*d_I+24));
    set_output_multiple(8);
    d_decbits=new unsigned char[d_I+6]; // this must include the tail bits!!
    d_bm=new float[(d_I+6)*d_fsm.O()];
    d_ibuf = new float[d_I*4+24];
    d_icnt = d_ocnt = 0;
}

dabp_vitdec::~dabp_vitdec()
{
    delete [] d_decbits;
    delete [] d_bm;
}

void dabp_vitdec::reset(int I)
{
    assert(I%8==0);
    gruel::scoped_lock guard(d_mutex); // avoid simultaneous access from work()
    delete [] d_decbits;
    delete [] d_bm;
    delete [] d_ibuf;
    d_I = I;
    d_nbytes = d_I/8;
    d_decbits=new unsigned char[d_I+6];
    d_bm=new float[(d_I+6)*d_fsm.O()];
    d_ibuf = new float[d_I*4+24];
    d_icnt = d_ocnt = 0;
}

const float dabp_vitdec::INF=1.0e30;

/* This was adopted from the gr-trellis package in gnuradio
 * We might change to SSE accelerated version later
 */
void dabp_vitdec::viterbi_algorithm(int I, int S, int O, 
             const std::vector<int> &NS,
             const std::vector<int> &OS,
             const std::vector< std::vector<int> > &PS,
             const std::vector< std::vector<int> > &PI,
             int K,
             int S0,int SK,
             const float *in, unsigned char *out)
{
    std::vector<int> trace(S*K);
    std::vector<float> alpha(S*2);
    int alphai;
    float norm,mm,minm;
    int minmi;
    int st;
    
    if(S0<0) { // initial state not specified
        for(int i=0;i<S;i++) 
            alpha[0*S+i]=0;
    }
    else {
        for(int i=0;i<S;i++) 
            alpha[0*S+i]=INF;
        alpha[0*S+S0]=0.0;
    }
    
    alphai=0;
    for(int k=0;k<K;k++) {
        norm=INF;
        for(int j=0;j<S;j++) { // for each next state do ACS
            minm=INF;
            minmi=0;
            for(unsigned int i=0;i<PS[j].size();i++) {
                //int i0 = j*I+i;
                if((mm=alpha[alphai*S+PS[j][i]]+in[k*O+OS[PS[j][i]*I+PI[j][i]]])<minm)
                    minm=mm,minmi=i;
            }
            trace[k*S+j]=minmi;
            alpha[((alphai+1)%2)*S+j]=minm;
            if(minm<norm) norm=minm;
        }
        for(int j=0;j<S;j++) 
            alpha[((alphai+1)%2)*S+j]-=norm; // normalize total metrics so they do not explode
        alphai=(alphai+1)%2;
    }
    
    if(SK<0) { // final state not specified
        minm=INF;
        minmi=0;
        for(int i=0;i<S;i++)
            if((mm=alpha[alphai*S+i])<minm) 
                minm=mm,minmi=i;
        st=minmi;
    }
    else {
        st=SK;
    }
    
    for(int k=K-1;k>=0;k--) { // traceback
        int i0=trace[k*S+st];
        out[k]= (unsigned char) PI[st][i0];
        st=PS[st][i0];
    }
}

void dabp_vitdec::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%8==0);
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    int input_required=noutput_items*(4*d_I+24)/d_nbytes;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_vitdec::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%8==0);
    int i,j,l;
    const float *in0=(const float*)input_items[0];
    const char *in1=(const char*)input_items[1];
    unsigned char *out0=(unsigned char*)output_items[0];
    char *out1=(char*)output_items[1];
    
    int nconsumed=0, nproduced=0;
    // lock the mutex
    gruel::scoped_lock guard(d_mutex);
    const int ilen = 4*d_I+24, olen = d_nbytes;
    while(nconsumed<ninput_items[0] && nproduced<noutput_items) {
        // process input from in to ibuf
        if(d_icnt<ilen) { // ibuf not full, fill it
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
            if(d_icnt+ninput_items[0]-nconsumed<ilen) { // not enough to fill ibuf
                memcpy(d_ibuf+d_icnt, in0, (ninput_items[0]-nconsumed)*sizeof(float));
                d_icnt += ninput_items[0]-nconsumed;
                nconsumed = ninput_items[0];
                break;
            }else {
                memcpy(d_ibuf+d_icnt, in0, (ilen-d_icnt)*sizeof(float));
                in0 += ilen-d_icnt;
                in1 += ilen-d_icnt;
                nconsumed += ilen-d_icnt;
                d_icnt = ilen;
                
                // VA decode ibuf->obuf
                // prepare the branch metrics
                branch_metrics(d_ibuf,ilen,d_fsm,d_bm);
                // viterbi algorithm
                viterbi_algorithm(d_fsm.I(),d_fsm.S(),d_fsm.O(),d_fsm.NS(),d_fsm.OS(),d_fsm.PS(),d_fsm.PI(),d_I+6,0,0,d_bm,d_decbits);
            }
        }
        
        // from obuf to out
        assert(d_icnt==ilen);
        if(olen-d_ocnt>noutput_items-nproduced) { // the output buffer is too small
            bit2byte(d_decbits+d_ocnt*8, out0, noutput_items-nproduced);
            memset(out1, 0, noutput_items-nproduced);
            if(d_ocnt==0)
                out1[0]=1;
            d_ocnt += noutput_items-nproduced;
            nproduced = noutput_items;
            break;
        }else { // the output buffer can hold all data
            bit2byte(d_decbits+d_ocnt*8, out0, olen-d_ocnt);
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

void dabp_vitdec::branch_metrics(const float *x, int len, const fsm &f, float *bm)
{
    const int O=f.O(); 
    const int D=log2i(O); // output bits per trellis stage
    assert(len%D==0);
    const int K=len/D; // info bit length
    unsigned int i,j,k;
    for(i=0;i<(unsigned int)K;i++) { // for each trellis stage
        for(j=0;j<(unsigned int)O;j++) { // for each output branch
            bm[i*O+j]=0;
            for(k=0;k<(unsigned int)D;k++) { // for each output bit. 0->+1; 1->-1
                // because viterbi_algorithm finds minimum path metric, we invert the branch metrics
                bm[i*O+j] -= x[i*D+k] * (((j>>(D-k-1))&1)? -1.0:1.0);
            }
        }
    }
}

int dabp_vitdec::log2i(int x)
{
    if(x<=0)
        return -1;
    x--;
    int i;
    for(i=0;x;x>>=1) {
        i++;
    }
    return i;
}

void dabp_vitdec::bit2byte(const unsigned char *x, unsigned char *y, int nbytes)
{
    unsigned int i,j;
    for(i=0;i<(unsigned int)nbytes;i++) {
        y[i]=0;
        for(j=0;j<8;j++) {
            y[i]=(y[i]<<1)|(x[i*8+j]&1);
        }
    }
}

