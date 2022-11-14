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

void grgpu_d2h_c_cuda_work_device(int noutput_items, const unsigned long* input_items,float* output_items) 
{

  // pointer for device memory
  float * d_odata=(float*)input_items[0];
  printf("D2H input device pointer: %p\n", d_odata);
  grgpu_fifo_pop_host((unsigned long*)&(input_items[0]), (void*)output_items, noutput_items, sizeof(float)*2);
  cudaThreadSynchronize();

  //  cudaFree(d_odata); 
}
