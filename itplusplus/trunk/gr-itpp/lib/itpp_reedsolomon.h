/* -*- c++ -*- */
/*
 * Copyright 2010 Communications Engineering Lab, KIT
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
#ifndef INCLUDED_ITPP_REEDSOLOMON_H
#define INCLUDED_ITPP_REEDSOLOMON_H

#include <gr_sync_block.h>
#include <itpp/comm/reedsolomon.h>

/*********** Base class ****************************************************/
class itpp_reedsolomon_coder
{
 protected:
	int d_m;
	int d_n;
	int d_k;

	itpp::Reed_Solomon *d_reedsolomon_coder;

 public:
	itpp_reedsolomon_coder(int m, int t, bool systematic);
	~itpp_reedsolomon_coder() { delete d_reedsolomon_coder; };

	double get_rate() { return d_reedsolomon_coder->get_rate(); };
	int get_m() { return d_m; };
	int get_n() { return d_n; };
	int get_k() { return d_k; };
};


/*********** Encoder *******************************************************/
class itpp_reedsolomon_encoder_vbb;

typedef boost::shared_ptr<itpp_reedsolomon_encoder_vbb> itpp_reedsolomon_encoder_vbb_sptr;

itpp_reedsolomon_encoder_vbb_sptr
itpp_make_reedsolomon_encoder_vbb (int m, int t, bool systematic = false);

class itpp_reedsolomon_encoder_vbb : public gr_sync_block, public itpp_reedsolomon_coder
{
  private:
	friend itpp_reedsolomon_encoder_vbb_sptr itpp_make_reedsolomon_encoder_vbb (int m, int t, bool systematic);

	itpp_reedsolomon_encoder_vbb (int m, int t, bool systematic);

  public:
	~itpp_reedsolomon_encoder_vbb () {};

	int work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
};


/*********** Decoder *******************************************************/
class itpp_reedsolomon_decoder_vbb;

typedef boost::shared_ptr<itpp_reedsolomon_decoder_vbb> itpp_reedsolomon_decoder_vbb_sptr;

itpp_reedsolomon_decoder_vbb_sptr
itpp_make_reedsolomon_decoder_vbb (int m, int t, bool systematic = false);

class itpp_reedsolomon_decoder_vbb : public gr_sync_block, public itpp_reedsolomon_coder
{
  private:
	friend itpp_reedsolomon_decoder_vbb_sptr itpp_make_reedsolomon_decoder_vbb (int m, int t, bool systematic);

	itpp_reedsolomon_decoder_vbb (int m, int t, bool systematic);

  public:
	~itpp_reedsolomon_decoder_vbb () {};

	int work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
};

#endif /* INCLUDED_ITPP_REEDSOLOMON_H */
