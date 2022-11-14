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
#ifndef INCLUDED_ITPP_HAMMING_H
#define INCLUDED_ITPP_HAMMING_H

#include <gr_sync_block.h>
#include <itpp/comm/hammcode.h>

/************** Base class *************************************************/
class itpp_hamming_coder
{
 protected:
	itpp::Hamming_Code *d_hamming_coder;

 public:
	itpp_hamming_coder (short m) { d_hamming_coder = new itpp::Hamming_Code(m); };
	~itpp_hamming_coder () { delete d_hamming_coder; };

	double get_rate() { return d_hamming_coder->get_rate(); };
	short get_n() { return d_hamming_coder->get_n(); };
	short get_k() { return d_hamming_coder->get_k(); };
};


/************** Encoder ****************************************************/
class itpp_hamming_encoder_vbb;

typedef boost::shared_ptr<itpp_hamming_encoder_vbb> itpp_hamming_encoder_vbb_sptr;

itpp_hamming_encoder_vbb_sptr itpp_make_hamming_encoder_vbb (short m);

/**
 * \brief Hamming encoder. Input and output are vectors, one bit per byte.
 */
class itpp_hamming_encoder_vbb : public gr_sync_block, public itpp_hamming_coder
{
private:
  friend itpp_hamming_encoder_vbb_sptr itpp_make_hamming_encoder_vbb (short m);

  itpp_hamming_encoder_vbb (short m);

 public:
  ~itpp_hamming_encoder_vbb () {};

  int work (int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

/************** Decoder ****************************************************/
class itpp_hamming_decoder_vbb;

typedef boost::shared_ptr<itpp_hamming_decoder_vbb> itpp_hamming_decoder_vbb_sptr;

itpp_hamming_decoder_vbb_sptr itpp_make_hamming_decoder_vbb (short m);

/**
 * \brief Hamming decoder. Input and output are vectors, one bit per byte.
 */
class itpp_hamming_decoder_vbb : public gr_sync_block, public itpp_hamming_coder
{
private:
  friend itpp_hamming_decoder_vbb_sptr itpp_make_hamming_decoder_vbb (short m);

  itpp_hamming_decoder_vbb (short m);

 public:
  ~itpp_hamming_decoder_vbb () {};

  int work (int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_ITPP_HAMMING_H */
