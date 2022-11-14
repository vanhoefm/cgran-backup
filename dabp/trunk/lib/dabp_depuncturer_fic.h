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
#ifndef INCLUDED_DABP_DEPUNCTURER_FIC_H
#define INCLUDED_DABP_DEPUNCTURER_FIC_H
#include <gr_block.h>

class dabp_depuncturer_fic;

typedef boost::shared_ptr<dabp_depuncturer_fic> dabp_depuncturer_fic_sptr;

dabp_depuncturer_fic_sptr dabp_make_depuncturer_fic(int mode);

/*! \brief Depuncturer for FIC
 * One input stream: float FIC or MSC samples
 * One output stream: float depunctured samples, either FIC or MSC
 * The output stream is organized as one convolutional code word (4*I+24 soft bits) at a time
 * Thus, M softbits are consumed at a time
 */
class dabp_depuncturer_fic : public gr_block
{
    private:
    friend dabp_depuncturer_fic_sptr dabp_make_depuncturer_fic(int mode);
    dabp_depuncturer_fic(int mode);
    int d_mode;
    int d_I, d_depunct_len, d_M;
    int d_L1, d_L2;
    const char *d_PI1, *d_PI2;
    
    static const char VPI16[32], VPI15[32], VT[24];
    
    public:
    ~dabp_depuncturer_fic();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
    int getI() const { return d_I;}
};

#endif // INCLUDED_DABP_DEPUNCTURER_FIC_H

