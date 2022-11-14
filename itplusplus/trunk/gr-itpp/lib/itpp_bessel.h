/* -*- c++ -*- */
/*
 * Copyright 2010 Communications Engineering Lab, KIT
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the Gorder General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * Gorder General Public License for more details.
 * 
 * You should have received a copy of the Gorder General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */
#ifndef INCLUDED_ITPP_BESSEL_H
#define INCLUDED_ITPP_BESSEL_H

#include <gr_sync_block.h>

/************** Bessel function of first kind  ******************************/
class itpp_besselj_ff;

typedef boost::shared_ptr<itpp_besselj_ff> itpp_besselj_ff_sptr;

itpp_besselj_ff_sptr itpp_make_besselj_ff (int order = 0);

/**
 * \brief Calculate Bessel function of first kind.
 */
class itpp_besselj_ff : public gr_sync_block
{
private:
  friend itpp_besselj_ff_sptr itpp_make_besselj_ff (int order);

  itpp_besselj_ff (int order);

  int d_order;

 public:
  ~itpp_besselj_ff ();

  bool set_order(int order);
  int order() { return d_order; };

  int work (int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

/************** Bessel function of first kind  ******************************/
class itpp_besseli_ff;

typedef boost::shared_ptr<itpp_besseli_ff> itpp_besseli_ff_sptr;

itpp_besseli_ff_sptr itpp_make_besseli_ff (int order = 0);

/**
 * \brief Calculate modified Bessel function of first kind.
 */
class itpp_besseli_ff : public gr_sync_block
{
private:
  friend itpp_besseli_ff_sptr itpp_make_besseli_ff (int order);

  itpp_besseli_ff (int order);

  int d_order;

 public:
  ~itpp_besseli_ff ();

  bool set_order(int order);
  int order() { return d_order; };

  int work (int noutput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_ITPP_BESSEL_H */
