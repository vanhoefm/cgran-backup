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
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef INCLUDED_GR_OFDM_INSERT_PREAMBLE_H
#define INCLUDED_GR_OFDM_INSERT_PREAMBLE_H

#include <gr_block.h>
#include <vector>

class gr_ofdm_insert_preamble;
typedef boost::shared_ptr<gr_ofdm_insert_preamble> gr_ofdm_insert_preamble_sptr;

gr_ofdm_insert_preamble_sptr
gr_make_ofdm_insert_preamble(int fft_length,
			     const std::vector<std::vector<gr_complex> > &preamble);

/*!
 * \brief insert "pre-modulated" preamble symbols before each payload.
 * \ingroup sync_blk
 * \ingroup ofdm_blk
 *
 * <pre>
 * input 1: stream of vectors of gr_complex [fft_length]
 *          These are the modulated symbols of the payload.
 *
 * input 2: stream of char.  The LSB indicates whether the corresponding
 *          symbol on input 1 is the first symbol of the payload or not.
 *          It's a 1 if the corresponding symbol is the first symbol,
 *          otherwise 0.
 *
 * N.B., this implies that there must be at least 1 symbol in the payload.
 *
 *
 * output 1: stream of vectors of gr_complex [fft_length]
 *           These include the preamble symbols and the payload symbols.
 *
 * output 2: stream of char.  The LSB indicates whether the corresponding
 *           symbol on input 1 is the first symbol of a packet (i.e., the
 *           first symbol of the preamble.)   It's a 1 if the corresponding
 *           symbol is the first symbol, otherwise 0.
 * </pre>
 *
 * \param fft_length length of each symbol in samples.
 * \param preamble   vector of symbols that represent the pre-modulated preamble.
 */

class gr_ofdm_insert_preamble : public gr_block
{
  friend gr_ofdm_insert_preamble_sptr
  gr_make_ofdm_insert_preamble(int fft_length,
			       const std::vector<std::vector<gr_complex> > &preamble);

protected:
  gr_ofdm_insert_preamble(int fft_length,
			  const std::vector<std::vector<gr_complex> > &preamble);

private:
  enum state_t {
    ST_IDLE,
    ST_PREAMBLE,
    ST_FIRST_PAYLOAD,
    ST_PAYLOAD
  };

  int						d_fft_length;
  const std::vector<std::vector<gr_complex> > 	d_preamble;
  std::vector<std::vector<gr_complex> > 	d_preamble_nc; // linklab, preamble for non-contiguous OFDM
  state_t					d_state;
  int						d_nsymbols_output;
  int						d_pending_flag;

  void enter_idle();
  void enter_preamble();
  void enter_first_payload();
  void enter_payload();
  

public:
  ~gr_ofdm_insert_preamble();

  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
  
  // linklab, reset preamble
  void reset_preamble(std::vector<std::vector<gr_complex> > new_preamle);
};

#endif /* INCLUDED_GR_OFDM_INSERT_PREAMBLE_H */
