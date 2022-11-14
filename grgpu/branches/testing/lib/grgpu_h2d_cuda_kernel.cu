/* -*- cuda -*- */
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

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <cutil_inline.h>
#include <grgpu_utils.h>

void grgpu_h2d_cuda_init_device(grgpu_fifo**context) 
{
  grgpu_fifo *fifo = *context;
  if(fifo->buffer==0){
    //wrap_pad and history are in bytes
    //WLP - wrap_pad should be pegged to a GR internal variable
    fifo->wrap_pad = fifo->multiple*16*fifo->token_size+fifo->history;
    cudaMalloc(((void**)&(fifo->buffer)), fifo->token_size*(fifo->length+fifo->multiple));
    fifo->head = fifo->buffer+fifo->history;
    checkCUDAError("Malloc");
  }
}

void grgpu_h2d_cuda_work_device(int noutput_items, const float* input_items,unsigned long* output_items, grgpu_fifo**context)
{
  grgpu_h2d_cuda_init_device(context);
  
  //use grgpu to push the data
  grgpu_fifo_push(*context, (void*)input_items, output_items, noutput_items);
}
