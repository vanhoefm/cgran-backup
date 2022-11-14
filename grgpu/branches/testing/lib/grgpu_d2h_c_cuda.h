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
 */
#ifndef INCLUDED_GRGPU_D2H_C_CUDA_H
#define INCLUDED_GRGPU_D2H_C_CUDA_H

#include <gr_block.h>

class grgpu_d2h_c_cuda;

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
typedef boost::shared_ptr<grgpu_d2h_c_cuda> grgpu_d2h_c_cuda_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of grgpu_d2h_c_cuda.
 *
 * To avoid accidental use of raw pointers, grgpu_d2h_c_cuda's
 * constructor is private.  grgpu_make_d2h_c_cuda is the public
 * interface for creating new instances.
 */
grgpu_d2h_c_cuda_sptr grgpu_make_d2h_c_cuda ();

/*!
 * \brief convert device pointers to host data
 * \ingroup block
 *
 * \details From the host's perspective, the \c d2h converts a device pointer
 * which are the input and output of other \c grgpu actors to a host
 * data.  Behind the scenes, the \c d2h actor performs a device to
 * host copy of data to the appropriate output buffer, which is
 * currently restricted to be of type \c float.  The values of the
 * input data to the actor are device pointers, which have been cast
 * to \c unsigned \c longs. For performance reasons, the default
 * configuration of \c d2h is to read in the first value and the
 * number of output items and assume that outputs are contiguous in
 * device memory.  This is tied to the assumption that the \c STS is
 * being used with static device buffers that are manipulated in-place.
 */
class grgpu_d2h_c_cuda : public gr_block
{
private:
  // The friend declaration allows grgpu_make_d2h_c_cuda to
  // access the private constructor.

  friend grgpu_d2h_c_cuda_sptr grgpu_make_d2h_c_cuda ();

  grgpu_d2h_c_cuda ();  	// private constructor
  int d_verbose;

 public:
  ~grgpu_d2h_c_cuda ();	// public destructor
  void set_verbose (int verbose) { d_verbose = verbose; }


  // Where all the action really happens

  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_GRGPU_D2H_C_CUDA_H */
