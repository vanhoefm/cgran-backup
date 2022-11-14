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

void grgpu_h2d_c_cuda_init_device(void**context) 
{
  if(*context==0x0){
    (*context) = malloc(sizeof(grgpu_fifo));
    grgpu_fifo *fifo = (grgpu_fifo*)*context;
    fifo->length = 1024*256;
    fifo->history = 200*sizeof(float)*2;
    fifo->token_size = sizeof(float)*2;
    cudaMalloc(((void**)&(fifo->buffer)), fifo->token_size*(fifo->length+1024));
    fifo->head = fifo->buffer;
    checkCUDAError("Malloc");
  }
}

void grgpu_h2d_c_cuda_work_device(int noutput_items, const float* input_items,unsigned long* output_items, void**context)
{
  // first see if the object from which this call was made has
  // initiated its cuda context
  if(*context==0x0){
    grgpu_h2d_c_cuda_init_device(context);
  }
  //the only thing in the h2d context is a fifo
  grgpu_fifo *fifo;
  fifo = (grgpu_fifo*)*context;

  //use grgpu to push the data
  grgpu_fifo_push(fifo, (void*)input_items, output_items, noutput_items);

}
