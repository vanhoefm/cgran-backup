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

#include <gr_randomize_bb.h>
#include <gr_io_signature.h>
#include <stdexcept>

gr_randomize_bb_sptr
gr_make_randomize_bb (unsigned int tap_mask, unsigned int preload)
{
  return gr_randomize_bb_sptr (new gr_randomize_bb (tap_mask, preload));
}

gr_randomize_bb::gr_randomize_bb (unsigned int tap_mask, unsigned int preload)
  : gr_sync_block ("randomize_bb",
		   gr_make_io_signature (1, 1, sizeof (unsigned char)),
		   gr_make_io_signature (1, 1, sizeof (unsigned char))),
    d_shift_register(preload)
{
    // EXCEPTION TEST FOR tap_mask containing bit 0
    if( (tap_mask & 0x01) == 1)
      {
        fprintf(stderr, "randomize_b: tap_mask cannot contain bit 0).\n");
        throw std::invalid_argument ("randomize_b: tap_mask cannot contain bit 0).");
      }

    // Record which bits are set in the tap_mask
    d_tap_count = 0;
    for(int i=0; i<32; i++)
      {
        if(tap_mask&0x01 == 1)
          {
            d_taps[d_tap_count] = i;
            d_tap_count++;
          }
        tap_mask = tap_mask >> 1;
      }
}

int
gr_randomize_bb::work (int                       noutput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star       &output_items)
{
  const unsigned char* in  = (const unsigned char *) input_items[0];
  unsigned char*       out = (unsigned char *) output_items[0];
  unsigned char        unscrambled_bit;
  unsigned char        scrambled_bit;
  int                  tap_bit;

  for (int i = 0; i < noutput_items; i++)
    {
        // Get next bit
        unscrambled_bit = in[i] & 0x01;

        // Shift the shift_register left by one bit
        d_shift_register <<= 1;

        // Compute the XOR of the unscrambled bit and the selected tap bits.
        scrambled_bit = unscrambled_bit;
        for(int t=0; t<d_tap_count; t++)
          {
            tap_bit = (d_shift_register >> d_taps[t]) & 0x01;
            scrambled_bit = scrambled_bit ^ tap_bit;
          }

        // Feed scrambled bit back into shift register
        d_shift_register |= scrambled_bit;

        // Output the scrambled bit
        out[i] = scrambled_bit;
    }
  return noutput_items;
}
