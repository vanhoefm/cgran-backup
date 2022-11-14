/* -*- c++ -*- */
/*
 * Copyright 2011 FOI
 *
 * Copyright 2010 A.Kaszuba, R.Checinski, MUT
 *
 * This file is part of FOI-MIMO
 * 
 * FOI-MIMO is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * FOI-MIMO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FOI-MIMO; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_FOIMIMO_OFDM_ALAMOUTI_TX_CC_H
#define INCLUDED_FOIMIMO_OFDM_ALAMOUTI_TX_CC_H

#include <gr_sync_block.h>
#include <vector>

class foimimo_ofdm_alamouti_tx_cc;
typedef boost::shared_ptr<foimimo_ofdm_alamouti_tx_cc> foimimo_ofdm_alamouti_tx_cc_sptr;

foimimo_ofdm_alamouti_tx_cc_sptr
foimimo_make_ofdm_alamouti_tx_cc(int fft_length);

/*!
 * \brief Take in OFDM symbol and perform Alamouti space-time coding for transmit
 * \ingroup ofdm_blk
 *
 * \param fft_length length of each symbol in samples.
 */

class foimimo_ofdm_alamouti_tx_cc : public gr_sync_block
{
  friend foimimo_ofdm_alamouti_tx_cc_sptr
  foimimo_make_ofdm_alamouti_tx_cc(int fft_length);

protected:
  foimimo_ofdm_alamouti_tx_cc(int fft_length);

private:
  int d_fft_length;
public:
  ~foimimo_ofdm_alamouti_tx_cc();

  int work (int noutput_items,
	    gr_vector_const_void_star &input_items,
	    gr_vector_void_star &output_items); 
  
};

#endif /* INCLUDED_FOIMIMO_OFDM_ALAMOUTI_TX_CC_H */
