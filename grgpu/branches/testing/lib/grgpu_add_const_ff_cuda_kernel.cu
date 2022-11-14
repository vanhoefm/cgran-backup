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
#include <grgpu_add_const_ff_cuda_kernel.h>


__global__ void add_const_ff_cuda_kernel(float* d_idata, float *d_odata, float k, int n)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<n)
    d_odata[idx]=d_idata[idx]+k;
}


void grgpu_add_const_ff_cuda_work_device(int noutput_items, const unsigned long* input_items,unsigned long* output_items, grgpu_add_const_ff_cuda_context* context) 
{

  // pointer for device memory
  float *d_idata = (float*)input_items[0];
  float *d_odata;
  // Use in-place buffer for performance
  d_odata = (float*)input_items[0];
		
#define tpb 128
  int grid = noutput_items/tpb;
  if(noutput_items % tpb)
    grid++;
  dim3 dimGrid(grid);
  dim3 dimBlock(tpb);

  add_const_ff_cuda_kernel<<<dimGrid, dimBlock>>>(d_idata, d_odata, context->constant, noutput_items);
  checkCUDAError("kernel execution 1");

  //now fill out the output output items array with the corresponding device pointers
  for(int i=0; i<noutput_items; i++){
    output_items[i]=(unsigned long)d_odata+i*sizeof(float);
  }
}

