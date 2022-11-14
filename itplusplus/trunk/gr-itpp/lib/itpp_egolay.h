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
#ifndef INCLUDED_ITPP_EGOLAY_H
#define INCLUDED_ITPP_EGOLAY_H

#include <gr_sync_block.h>
#include <itpp/comm/egolay.h>

/*************** Base class *************************************************/
class itpp_egolay_coder
{
 protected:
	itpp::Extended_Golay *d_egolay_coder;

 public:
	itpp_egolay_coder() { d_egolay_coder = new itpp::Extended_Golay(); };
	~itpp_egolay_coder() { delete d_egolay_coder; };

	double get_rate() { return 0.5; };
};


/*************** Encoder ****************************************************/
class itpp_egolay_encoder_vbb;

typedef boost::shared_ptr<itpp_egolay_encoder_vbb> itpp_egolay_encoder_vbb_sptr;

itpp_egolay_encoder_vbb_sptr itpp_make_egolay_encoder_vbb ();

/**
 * \brief Extended (24, 12, 8) Golay code encoder.
 */
class itpp_egolay_encoder_vbb : public gr_sync_block, public itpp_egolay_coder
{
  private:
	friend itpp_egolay_encoder_vbb_sptr itpp_make_egolay_encoder_vbb ();

	itpp_egolay_encoder_vbb ();

  public:
	~itpp_egolay_encoder_vbb () {};

	int work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
};


/*************** Decoder ****************************************************/
class itpp_egolay_decoder_vbb;

typedef boost::shared_ptr<itpp_egolay_decoder_vbb> itpp_egolay_decoder_vbb_sptr;

itpp_egolay_decoder_vbb_sptr itpp_make_egolay_decoder_vbb ();

/**
 * \brief Extended (24, 12, 8) Golay code decoder.
 */
class itpp_egolay_decoder_vbb : public gr_sync_block, public itpp_egolay_coder
{
  private:
	friend itpp_egolay_decoder_vbb_sptr itpp_make_egolay_decoder_vbb ();

	itpp_egolay_decoder_vbb ();

  public:
	~itpp_egolay_decoder_vbb () {};

	int work (int noutput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
};

#endif /* INCLUDED_ITPP_EGOLAY_H */
