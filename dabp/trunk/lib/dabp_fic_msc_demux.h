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

#ifndef INCLUDED_DABP_FIC_MSC_DEMUX_H
#define INCLUDED_DABP_FIC_MSC_DEMUX_H
#include <gr_block.h>

class dabp_fic_msc_demux;

typedef boost::shared_ptr<dabp_fic_msc_demux> dabp_fic_msc_demux_sptr;
/*! 
 * \param sel select FIC (sel=0) or MSC (sel=1)
 * \param mode DAB mode
 */
dabp_fic_msc_demux_sptr dabp_make_fic_msc_demux(int sel, int mode=1);

/*! \brief FIC and MSC demultiplexer
 * One input stream: float demodulated samples
 * One output stream: float demultiplexed samples, either FIC or MSC
 * The output stream is organized as one DAB frame at a time
 */
class dabp_fic_msc_demux : public gr_block
{
    private:
    friend dabp_fic_msc_demux_sptr dabp_make_fic_msc_demux(int sel, int mode);
    dabp_fic_msc_demux(int sel, int mode=1);
    int d_sel, d_mode, d_K;
    int d_fic_syms, d_msc_syms; // symbols per frame for FIC/MSC
    int d_syms_per_frame; // symbols per frame for selected channel
    public:
    ~dabp_fic_msc_demux();
	void forecast (int noutput_items, gr_vector_int &ninput_items_required);
    int general_work (int noutput_items,
                gr_vector_int &ninput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items);
};
#endif // INCLUDED_DABP_FIC_MSC_DEMUX_H

