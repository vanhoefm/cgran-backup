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
#ifndef INCLUDED_DABP_SCRAMBLER_H
#define INCLUDED_DABP_SCRAMBLER_H
#include <gr_sync_block.h>
#include <boost/thread.hpp>

class dabp_scrambler;

typedef boost::shared_ptr<dabp_scrambler> dabp_scrambler_sptr;

dabp_scrambler_sptr dabp_make_scrambler(int I);

/*! \brief DAB scrambler (energy dispersal)
 * It takes in I/8 bytes at a time
 */
class dabp_scrambler : public gr_block
{
    private:
    friend dabp_scrambler_sptr dabp_make_scrambler(int I);
    dabp_scrambler(int I);
    void init_tab();
    void init_prbs();
    int d_nbytes; // I/8 the number of bytes corrsponding to the length of PRBS
    unsigned short tab[512]; // prbs table
    unsigned char *prbs; // the precalculated PRBS
    unsigned short run8(unsigned char regs[]);
    
    boost::mutex d_mutex;
    unsigned char * d_ibuf;
    int d_icnt, d_ocnt;
    
    public:
    ~dabp_scrambler();
    void forecast(int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
    int get_nbytes() const { return d_nbytes; }
    void reset(int I);
};
#endif // INCLUDED_DABP_SCRAMBLER_H

