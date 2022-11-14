/* -*- c++ -*- */
/*
 * Copyright 2011 Free Software Foundation, Inc.
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
 *
 * This file was modified by William Plishker in 2011 for the GNU Radio 
 * support package GRGPU.  See www.cgran.org/wiki/GRGPU for more details. 
 */
#ifndef INCLUDED_GRGPU_H2D_CUDA_H
#define INCLUDED_GRGPU_H2D_CUDA_H

#include <gr_block.h>
#include <grgpu_utils.h>

class grgpu_h2d_cuda;

/*
 * We use boost::shared_ptr's instead of raw pointers for all access
 * to gr_blocks (and many other data structures).  The shared_ptr gets
 * us transparent reference counting, which greatly simplifies storage
 * management issues.  This is especially helpful in our hybrid
 * C++ / Python system.
 *
 * See http://www.boost.org/libs/smart_ptr/smart_ptr.htm
 *
 * As a convention, the _sptr suffix indicates a boost::shared_ptr
 */
typedef boost::shared_ptr<grgpu_h2d_cuda> grgpu_h2d_cuda_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of grgpu_h2d_cuda.
 *
 * To avoid accidental use of raw pointers, grgpu_h2d_cuda's
 * constructor is private.  grgpu_make_h2d_cuda is the public
 * interface for creating new instances.
 */
grgpu_h2d_cuda_sptr grgpu_make_h2d_cuda ();

/*!
 * \brief convert host data to device pointers
 * \ingroup block
 *
 * \details From the host's perspective, the \c h2d converts host data
 * to device pointers, which are the input and output of other \c
 * grgpu actors.  Behind the scenes, the \c h2d actor performs a host
 * to device copy of data to a statically allocated, fixed size buffer
 * on the GPU.  Currently data is restricted to be of type \c float.
 * The values of the output data of the actor are device pointers,
 * which have been cast to \c unsigned \c longs. For performance
 * reasons, the default configuration of \c h2d is to return only the
 * pointer to the first value so that subsequent actors must assume that outputs
 * are contiguous in device memory.  This is tied to the assumption
 * that the \c STS is being used with static device buffers that are
 * manipulated in-place.
 */
class grgpu_h2d_cuda : public gr_block
{
private:
  // The friend declaration allows grgpu_make_h2d_cuda to
  // access the private constructor.

  friend grgpu_h2d_cuda_sptr grgpu_make_h2d_cuda ();

  grgpu_h2d_cuda ();  	// private constructor
  int verbose;
  int hist;
  int multiple;
  int length;
  grgpu_fifo **context;

 public:
  ~grgpu_h2d_cuda ();	// public destructor
  void set_verbose (int v)     { verbose = v;}
  void set_length (int n)      { length = n;}
  void set_output_mul(int mul) { multiple = mul; this->set_output_multiple(mul);}
  void set_history(int n)      { hist = n;}

  // Where all the action really happens

  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_GRGPU_H2D_CUDA_H */
