/* -*- c++ -*- */
/*
 * Copyright 2007 Free Software Foundation, Inc.
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

GR_SWIG_BLOCK_MAGIC(gr,ofdm_insert_preamble);

gr_ofdm_insert_preamble_sptr
gr_make_ofdm_insert_preamble(int fft_length,
			     const std::vector<std::vector<gr_complex> > &preamble);


class gr_ofdm_insert_preamble : public gr_block
{
 protected:
  gr_ofdm_insert_preamble(int fft_length,
			  const std::vector<std::vector<gr_complex> > &preamble);
 public: 
  // linklab, reset preamble
  void reset_preamble(std::vector<std::vector<gr_complex> > new_preamle);
};
