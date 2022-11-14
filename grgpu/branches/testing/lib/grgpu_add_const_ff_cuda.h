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

#ifndef INCLUDED_GRGPU_ADD_CONST_FF_CUDA_H
#define INCLUDED_GRGPU_ADD_CONST_FF_CUDA_H

#include <gr_block.h>
#include <grgpu_add_const_ff_cuda_kernel.h>

class grgpu_add_const_ff_cuda;

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
typedef boost::shared_ptr<grgpu_add_const_ff_cuda> grgpu_add_const_ff_cuda_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of grgpu_add_const_ff_cuda.
 *
 * To avoid accidental use of raw pointers, grgpu_add_const_ff_cuda's
 * constructor is private.  grgpu_make_add_const_ff_cuda is the public
 * interface for creating new instances.
 */
grgpu_add_const_ff_cuda_sptr grgpu_make_add_const_ff_cuda (float);

/*!
 * \brief add a stream of floats
 * \ingroup block
 *
 */
class grgpu_add_const_ff_cuda : public gr_block
{
private:
  // The friend declaration allows grgpu_make_add_const_ff_cuda to
  // access the private constructor.

  friend grgpu_add_const_ff_cuda_sptr grgpu_make_add_const_ff_cuda (float);
  grgpu_add_const_ff_cuda_context * context;

  grgpu_add_const_ff_cuda (float);  	// private constructor

 public:
  ~grgpu_add_const_ff_cuda ();	// public destructor

  // Where all the action really happens

  int general_work (int noutput_items,
		    gr_vector_int &ninput_items,
		    gr_vector_const_void_star &input_items,
		    gr_vector_void_star &output_items);
};

#endif /* INCLUDED_GRGPU_ADD_CONST_FF_CUDA_H */
