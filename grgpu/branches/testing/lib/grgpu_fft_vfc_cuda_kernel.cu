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
 *
 * This file was modified by William Plishker in 2011 for the GNU Radio 
 * support package GRGPU.  See www.cgran.org/wiki/GRGPU for more details. 
 */ 

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <cutil_inline.h>
#include <grgpu_utils.h>
#include <cufft.h>


__global__ void fft_vfc_cuda_kernel(float* d_idata, float *d_odata, float k, int n)
{
  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  if(idx<n)
    d_odata[idx]=d_idata[idx]+k;
}


void grgpu_fft_vfc_cuda_work_device(int noutput_items, const unsigned long* input_items,unsigned long* output_items, void **plan) 
{

  if(*plan==0x0){
    *plan = malloc(sizeof(cufftHandle));
    cufftPlan1d((cufftHandle*)*plan, 1024, CUFFT_R2C, 1);
  }


  // pointer for device memory
  float *d_idata = (float*)input_items[0];
  float *d_odata;
  //  unsigned int o_size = (noutput_items) * sizeof (float);

  // Use in-place buffer for performance
  d_odata = (float*)input_items[0];
  //  cudaMalloc( (void **) &d_odata, o_size);
  //  checkCUDAError("Malloc");

  cufftExecR2C(*(cufftHandle*)*plan, (cufftReal*)d_idata, (cufftComplex*)d_odata);

  printf("%p\n", d_odata);
  //now fill out the output output items array with the corresponding device pointers
  for(int i=0; i<noutput_items; i+=1024){
    output_items[i]=(unsigned long)d_odata+i*8;
  }
}
