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
#ifndef INCLUDED_DABP_VITDEC_H
#define INCLUDED_DABP_VITDEC_H
#include <gr_block.h>
#include <fsm.h>
#include <boost/thread.hpp>

class dabp_vitdec;

typedef boost::shared_ptr<dabp_vitdec> dabp_vitdec_sptr;

dabp_vitdec_sptr dabp_make_vitdec(int I);

/*! Viterbi decoder for DAB
 * It takes 4*I+24 soft bits and produces I/8 bytes at a time
 * It assumes terminated trellis
 */
class dabp_vitdec : public gr_block
{
    private:
    friend dabp_vitdec_sptr dabp_make_vitdec(int I);
    dabp_vitdec(int I);
    int d_I;
    int d_nbytes; // =ceil(d_I/8)
    unsigned char *d_decbits; // decoded bits
    
    static const int GENERATOR[4];
    fsm d_fsm; // the Finite State Machine representing the trellis
    static const float INF;
    float *d_bm; // branch metrics
    
    static void branch_metrics(const float *x, int len, const fsm &f, float *bm);
    static void viterbi_algorithm(int I, int S, int O, 
             const std::vector<int> &NS,
             const std::vector<int> &OS,
             const std::vector< std::vector<int> > &PS,
             const std::vector< std::vector<int> > &PI,
             int K,
             int S0,int SK,
             const float *in, unsigned char *out);
    static int log2i(int x); // ceil(log2(x))
    static void bit2byte(const unsigned char *x, unsigned char *y, int nbytes);
    
    boost::mutex d_mutex;
    float * d_ibuf;
    int d_icnt, d_ocnt;
    
    public:
    ~dabp_vitdec();
    void reset(int I);
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};

#endif // INCLUDED_DABP_VITDEC_H

