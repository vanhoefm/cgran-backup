/* -*- c++ -*- */
/*
 * Copyright 2008 Free Software Foundation, Inc.
 * 
 * This file is part of GNU Radio
 * 
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
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

#include <gr_nrzi_to_nrz_bb.h>
#include <gr_io_signature.h>
#include <stdexcept>

gr_nrzi_to_nrz_bb_sptr
gr_make_nrzi_to_nrz_bb (bool preload)
{
  return gr_nrzi_to_nrz_bb_sptr (new gr_nrzi_to_nrz_bb (preload));
}

gr_nrzi_to_nrz_bb::gr_nrzi_to_nrz_bb (bool preload)
  : gr_sync_block ("nrzi_to_nrz_bb",
		   gr_make_io_signature (1, 1, sizeof (unsigned char)),
		   gr_make_io_signature (1, 1, sizeof (unsigned char))),
           d_prev_nrzi_bit(preload)
{
}

int
gr_nrzi_to_nrz_bb::work (int                       noutput_items,
			 gr_vector_const_void_star &input_items,
			 gr_vector_void_star       &output_items)
{
  const unsigned char* in  = (const unsigned char *) input_items[0];
  unsigned char*       out = (unsigned char *) output_items[0];
  unsigned char        nrzi_bit;
  unsigned char        nrz_bit;

  for (int i = 0; i < noutput_items; i++)
    {
        nrzi_bit = in[i];
        // Convert NRZI to NRZ.
        if(nrzi_bit != d_prev_nrzi_bit)
          {
            nrz_bit = 0;
          }
        else
          {
            nrz_bit = 1;
          }
        out[i] = nrz_bit;
        d_prev_nrzi_bit = nrzi_bit;
    }
  return noutput_items;
}
