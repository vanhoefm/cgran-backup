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

#include "dabp_fic_msc_demux.h"
#include <gr_io_signature.h>
#include <cassert>
#include <iostream>

dabp_fic_msc_demux_sptr dabp_make_fic_msc_demux(int sel, int mode)
{
    return dabp_fic_msc_demux_sptr(new dabp_fic_msc_demux(sel, mode));
}

dabp_fic_msc_demux::dabp_fic_msc_demux(int sel, int mode)
    : gr_block("fic_msc_demux", 
        gr_make_io_signature(1,1,sizeof(float)), 
        gr_make_io_signature(1,1,sizeof(float))), d_sel(sel), d_mode(mode)
{
    switch(mode) {
        case 1:
        d_K=1536;
        d_fic_syms=3;
        d_msc_syms=72;
        break;
        
        case 2:
        d_K=384;
        d_fic_syms=3;
        d_msc_syms=72;
        break;
        
        case 3:
        d_K=192;
        d_fic_syms=8;
        d_msc_syms=144;
        break;
        
        case 4:
        d_K=768;
        d_fic_syms=3;
        d_msc_syms=72;
        break;
        
        default:
        std::cerr<<"Unkown transmission mode-"<<mode<<std::endl;
        assert(false);
    }
    
    switch(sel){
        case 0: // FIC
        d_syms_per_frame = d_fic_syms;
        break;
        case 1: // MSC
        d_syms_per_frame = d_msc_syms;
        break;
        default:
        std::cerr<<"Unkown channel selection: "<<sel<<std::endl;
        assert(false);
    }
    set_relative_rate((float)d_syms_per_frame/(d_fic_syms+d_msc_syms));
    set_output_multiple(d_syms_per_frame*d_K*2);
}

dabp_fic_msc_demux::~dabp_fic_msc_demux()
{
}

void dabp_fic_msc_demux::forecast(int noutput_items, gr_vector_int &ninput_items_required)
{
    assert(noutput_items%(d_syms_per_frame*d_K*2)==0);
    int nblocks=noutput_items/(d_syms_per_frame*d_K*2);
    int input_required=(d_fic_syms+d_msc_syms)*d_K*2*nblocks;
    unsigned ninputs=ninput_items_required.size();
    for(unsigned i=0;i<ninputs;i++)
        ninput_items_required[i]=input_required;
}

int dabp_fic_msc_demux::general_work(int noutput_items,
                                gr_vector_int &ninput_items,
                                gr_vector_const_void_star &input_items,
                                gr_vector_void_star &output_items)
{
    assert(noutput_items%(d_syms_per_frame*d_K*2)==0);
    int nblocks=noutput_items/(d_syms_per_frame*d_K*2);
    const float *in=(const float*)input_items[0];
    float *out=(float*)output_items[0];
    int start= (d_sel==0) ? 0 : (d_fic_syms*d_K*2); // start index for selected channel
    for(int i=0;i<nblocks;i++) {
        for(int j=0;j<d_syms_per_frame*d_K*2;j++) 
			out[i*d_syms_per_frame*d_K*2+j]=in[i*(d_fic_syms+d_msc_syms)*d_K*2+j+start];
    }
    consume_each(nblocks*(d_fic_syms+d_msc_syms)*d_K*2);
    return noutput_items;
}

