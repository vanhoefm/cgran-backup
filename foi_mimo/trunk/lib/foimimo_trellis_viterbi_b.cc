/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
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
// This is a modification of trellis_viterbi_b.cc from GNU Radio.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <foimimo_trellis_viterbi_b.h>
#include <gr_io_signature.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
  
static const float INF = 1.0e9;

foimimo_trellis_viterbi_b_sptr
foimimo_make_trellis_viterbi_b (const fsm &FSM,int K,int S0,int SK)
{
  return foimimo_trellis_viterbi_b_sptr (new foimimo_trellis_viterbi_b (FSM,K,S0,SK));
}

foimimo_trellis_viterbi_b::foimimo_trellis_viterbi_b (const fsm &FSM,int K,int S0,int SK)
  : gr_block ("viterbi_b",
      gr_make_io_signature2 (2, 2, sizeof (float),sizeof(char)),
      gr_make_io_signature2 (2, 2, sizeof (unsigned char),sizeof(char))),
  d_FSM (FSM),
  d_K (K),
  d_S0 (S0),
  d_default_state(d_S0),
  d_SK (SK)
{
    set_relative_rate (1.0 / ((double) d_FSM.O()));
    set_output_multiple (d_K);
}


void
foimimo_trellis_viterbi_b::forecast (int noutput_items, gr_vector_int &ninput_items_required)
{
  assert (noutput_items % d_K == 0);
  int input_required =  d_FSM.O() * noutput_items ;
  unsigned ninputs = ninput_items_required.size();
  for (unsigned int i = 0; i < ninputs; i++) {
    ninput_items_required[i] = input_required;
  }
}

void
viterbi_algorithm(int I, int S, int O,
             const std::vector<int> &NS,
             const std::vector<int> &OS,
             const std::vector< std::vector<int> > &PS,
             const std::vector< std::vector<int> > &PI,
             int K,
             int S0,int SK,
             const float *in, unsigned char *out, unsigned char *new_pkt)
{
  std::vector<int> trace(S*K);
  std::vector<float> alpha(S*2);
  int alphai;
  float norm,mm,minm;
  int minmi;
  int st;


  if(S0<0) { // initial state not specified
      for(int i=0;i<S;i++) alpha[0*S+i]=0;
  }
  else {
      for(int i=0;i<S;i++) alpha[0*S+i]=INF;
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
          if((mm=alpha[alphai*S+i])<minm) minm=mm,minmi=i;
      st=minmi;
  }
  else {
      st=SK;
  }

  for(int k=K-1;k>=0;k--) { // traceback
      int i0=trace[k*S+st];
      out[k]= (unsigned char) PI[st][i0];
      new_pkt[k] = 0;
      st=PS[st][i0];
  }

}

void
foimimo_trellis_viterbi_b::set_K(int K)
{
  d_K = K;
  set_output_multiple (d_K);
}

int
foimimo_trellis_viterbi_b::general_work (int noutput_items,
                        gr_vector_int &ninput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items)
{
  const float *in = (const float *) input_items[0];
  const unsigned char *in_new_pkt = (unsigned char*) input_items[1];
  unsigned char *out = (unsigned char *) output_items[0];
  unsigned char *out_new_pkt = (unsigned char *) output_items[1];
  assert (noutput_items % d_K == 0);
  assert(d_FSM.O() * noutput_items <= ninput_items[0]);

  int nblocks = noutput_items / d_K;

  for (int n=0;n<nblocks;n++) {
    int packet_start_i = n*d_K*d_FSM.O();

    int i=packet_start_i + 1;
    while (i < d_K*d_FSM.O()){
      if (in_new_pkt[i] == 1){
        consume_each(i+packet_start_i);
        return(n*d_K);
      }
      i++;
    }

    viterbi_algorithm(d_FSM.I(),d_FSM.S(),d_FSM.O(),d_FSM.NS(),d_FSM.OS(),d_FSM.PS(),d_FSM.PI(),d_K,d_S0,d_SK,&(in[n*d_K*d_FSM.O()]),&(out[n*d_K]), &(out_new_pkt[n*d_K]));

    d_S0 = d_default_state;
    out_new_pkt[(n+1)*d_K-1]= 1;
  }
  consume_each (d_FSM.O() * noutput_items );

  return noutput_items;
}
