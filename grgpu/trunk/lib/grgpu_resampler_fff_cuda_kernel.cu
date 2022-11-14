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

	// Copy to Device Memory
typedef struct{
  float *d_taps;
  float *d_first_time_buff;
  grgpu_fifo *fifo;
  int upsample_rate;
  int downsample_rate;
  int num_of_taps;
  int start_point;
} grgpu_resampler_fff_context;

	


void grgpu_resampler_fff_cuda_init_device(void**context, float *h_taps) 
{
  if(*context==0x0){  // TODO Make this programmatic
    (*context) = malloc(sizeof(grgpu_resampler_fff_context));
    grgpu_resampler_fff_context * rc = (grgpu_resampler_fff_context*)(*context);
    rc->num_of_taps = 201;
    cutilSafeCall(cudaMalloc( (void**) &rc->d_taps, rc->num_of_taps*sizeof(float)));
    rc->upsample_rate = 10;
    rc->downsample_rate = 7;
    rc->start_point = 0;

    rc->fifo  = (grgpu_fifo *) malloc(sizeof(grgpu_fifo));
    grgpu_fifo *fifo = rc->fifo;
    fifo->length = 512*1024;//*rc->upsample_rate;
    fifo->history = 0;
    fifo->wrap_pad = 16000;
    fifo->token_size = sizeof(float);
    cudaMalloc(((void**)&(fifo->buffer)), fifo->token_size*(fifo->length));
    fifo->head = fifo->buffer;
    checkCUDAError("Malloc");

    cutilSafeCall(cudaMemcpy( rc->d_taps, h_taps, rc->num_of_taps*sizeof(float), cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMalloc( (void**) &(rc->d_first_time_buff), (rc->num_of_taps+1024*rc->upsample_rate)*sizeof(float)));
  }
}
	

__global__ void grgpu_resampler_fff_cuda_kernel( float* d_idata, float* d_odata, float* d_taps, int UPSAMPLE_RATE, int DOWNSAMPLE_RATE, int NUM_OF_TAPS, int DATA_OUT_LENGTH, int start_point)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int x = i * DOWNSAMPLE_RATE + start_point+(NUM_OF_TAPS-1)*UPSAMPLE_RATE;
	
	float sum = 0;
	
	if (i < DATA_OUT_LENGTH)
	{
		for(int j = x - x % UPSAMPLE_RATE; j > x-NUM_OF_TAPS ; j = j-UPSAMPLE_RATE)
		{
			sum = sum + d_idata[j/UPSAMPLE_RATE] * d_taps[x-j];
		}
		
		d_odata[i] = sum;
	}
	

	
}


void grgpu_resampler_fff_cuda_work_device(int noutput_items, const unsigned long* input_items,unsigned long* output_items, void**context, float* h_taps, int ninput_items) 
{
  // pointer for device memory
  float *d_idata = (float *) grgpu_fifo_pop_device((unsigned long*)&(input_items[0]), ninput_items, sizeof(float));
  float *d_odata;
  grgpu_resampler_fff_context * rc;
  //  unsigned int o_size = (noutput_items) * sizeof (float);
  if(*context==0x0){
    printf("init\n");
    grgpu_resampler_fff_cuda_init_device(context, h_taps);
    rc = (grgpu_resampler_fff_context*)(*context);
    float *zeros;
    zeros = (float *) malloc((rc->num_of_taps-1)*sizeof(float));
    for(int i=0; i<rc->num_of_taps-1; i++)
      zeros[i]=0.0;
    cutilSafeCall(cudaMemcpy( rc->d_first_time_buff, zeros, (rc->num_of_taps-1)*sizeof(float), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy( (void*)&(rc->d_first_time_buff[rc->num_of_taps-1]), (void*)d_idata, (ninput_items)*sizeof(float), cudaMemcpyDeviceToDevice));
    d_idata = rc->d_first_time_buff;
  } else {
    rc = (grgpu_resampler_fff_context *) *context;
    d_idata = (float*)(((unsigned long)d_idata) - (rc->num_of_taps-1)*rc->fifo->token_size);
  }


  d_odata = (float*)rc->fifo->head;
  unsigned long orig_head = rc->fifo->head;
  //  cudaMalloc( (void **) &d_odata, o_size);
  //  checkCUDAError("Malloc");
		
#define tpb 128
  int grid = noutput_items/tpb;
  if(noutput_items % tpb)
    grid++;
  dim3 dimGrid(grid);
  dim3 dimBlock(tpb);
  
  printf("kernel call %d, %d, %d, %d, %d\n", rc->upsample_rate, rc->downsample_rate, rc->num_of_taps, noutput_items, rc->start_point);
  grgpu_resampler_fff_cuda_kernel<<<noutput_items / 256 + 1, 256>>>( d_idata, d_odata, rc->d_taps, rc->upsample_rate, rc->downsample_rate, rc->num_of_taps, noutput_items, rc->start_point);
  
  cudaThreadSynchronize();
  
  checkCUDAError("kernel execution 1");

  rc->start_point = (noutput_items-rc->start_point)%rc->downsample_rate;
  rc->fifo->head = rc->fifo->head + noutput_items*rc->fifo->token_size;

  printf("d_odata: %p\n", d_odata);

  //check taps
  float taps[300];
  cutilSafeCall(cudaMemcpy( taps, rc->d_taps, rc->num_of_taps*sizeof(float), cudaMemcpyDeviceToHost));
  checkCUDAError("tap check");
  for(int j=0; j<201; j++){
    if(taps[j]-h_taps[j]>0.001){
      printf("Uh oh: %d, orig: %f, new %f\n", j, h_taps[j], taps[j]);
    }
  }

  //  cudaFree(d_idata);

  //now fill out the output output items array with the corresponding device pointers
  for(int i=0; i<noutput_items; i++){
    output_items[i]=orig_head+i*rc->fifo->token_size;
  }
}
