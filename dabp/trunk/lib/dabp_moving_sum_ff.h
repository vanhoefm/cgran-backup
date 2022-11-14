/* -*- c++ -*- */
/*
 * Copyright 2004 Free Software Foundation, Inc.
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
#ifndef INCLUDED_DABP_MOVING_SUM_FF_H
#define INCLUDED_DABP_MOVING_SUM_FF_H

#include <gr_sync_block.h>

class dabp_moving_sum_ff;

typedef boost::shared_ptr<dabp_moving_sum_ff> dabp_moving_sum_ff_sptr;

dabp_moving_sum_ff_sptr dabp_make_moving_sum_ff (int length, double alpha=0.9999);
/*!
 * \brief Moving sum over a stream of floats.
 * \ingroup filter
 * \param length length of the moving sum (=number of taps)
 *
 * input: float
 * output: float
 *
 * We use an IIR filter to approximate the moving sum 
 * A decay factor alpha<1 (default alpha=0.9999) is used to make sure the filter is stable
 */
class dabp_moving_sum_ff : public gr_sync_block
{
private:
  
  friend dabp_moving_sum_ff_sptr dabp_make_moving_sum_ff (int length, double alpha);

  dabp_moving_sum_ff (int length, double alpha=0.9999);    // private constructor

  double d_sum;
  int d_length;
  double d_alpha, d_alphapwr;
 public:
  ~dabp_moving_sum_ff ();  // public destructor
  int length() const {return d_length;}
  void reset() {d_sum=0;}

  int work (int noutput_items,
            gr_vector_const_void_star &input_items,
            gr_vector_void_star &output_items);
};

#endif /* INCLUDED_DABP_MOVING_SUM_FF_H */
