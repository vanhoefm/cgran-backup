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
 * This file was created by William Plishker in 2011 for the GNU Radio 
 * support package GRGPU.  See www.cgran.org/wiki/GRGPU for more details.
 */

#include <stdio.h>
#include <grgpu_utils.h>


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}


void grgpu_fifo_push(grgpu_fifo *fifo, void*input, unsigned long * output, int length) {
  unsigned long d_head     = fifo->head;
  unsigned long d_fifo_end = fifo->buffer+fifo->length*fifo->token_size;
  unsigned long h_input = (unsigned long) input;  
  unsigned long d_wrap_head = (fifo->wrap_pad - (d_fifo_end - d_head))+fifo->buffer;

  int i = length*fifo->token_size;
  printf("H2D: head %p, end %p, new end %p\n", d_head,d_fifo_end, d_head + i);

  /* before doing anything, check if we should be wrapping back*/ 
  if(d_head+i >= d_fifo_end){
    //  printf("H2D: writing history from %p (%d) to %p\n", fifo->head-fifo->history, fifo->history, fifo->buffer);
    /* move the history in front */
    /*cudaMemcpy((void*)(fifo->buffer), 
	       (void*)(fifo->head-fifo->history), 
	       fifo->history, 
	       cudaMemcpyDeviceToDevice);*/
    /* now we can reset the head pointer */
    // we want to wrap to the wrap_pad head
    d_head = d_wrap_head;
    printf("H2D: writing new entry to wrapped location %p\n", d_head);
  }

  //push the data to head
  printf("H2D: writing to %p, for %d (%d)\n", d_head, i, length);
  cudaMemcpy((void*)d_head, (void*)(h_input), i, cudaMemcpyHostToDevice);
  char ss[256];
  sprintf(ss, "H2D: Memcpy idata: %p", d_head);
  checkCUDAError(ss);

  /*now check if we're in the wrap pad.  If so, we'll keep a redundant
    copy of the end of the array in the beginning of the array (after
    the history pad)*/
/* -----------------------------------------------------------------------------------
   | redundant wrap_pad + history | free elements | prev elements | head | free space|
*/
  //if the current write to the buffer is in or goes into the wrap pad, 
  if(d_head+i > d_fifo_end - fifo->wrap_pad){
    unsigned long d_wrap_src = d_head;
    int wrap_length = i;
    // adjust the copy length if only part of the buffer enter the wrap pad
    if(d_wrap_head < fifo->buffer){
      wrap_length = i - (fifo->buffer - d_wrap_head);      
      d_wrap_head = fifo->buffer;
      d_wrap_src = d_head + (fifo->buffer - d_wrap_head);
    }
    printf("H2D: writing wrap_pad to %p from %p, for %d (%d), wrap_pad=%d\n", d_wrap_head, d_wrap_src, wrap_length, length, fifo->wrap_pad);
    cudaMemcpy((void*)d_wrap_head, (void*)(d_wrap_src), wrap_length, cudaMemcpyDeviceToDevice);
  }
  // now fill out the output output items array with the corresponding
  // device pointers.  As a performance optimization, we plan for the 
  // normal case to need only the first pointer and the buffer size for 
  // one contigous read. When a buffer wrap does occur, its detected by 
  // by noting a non-monotic pointer address in output_items.
  for(int ii=0; ii<i/fifo->token_size; ii++){
    unsigned long d_pointer = (unsigned long)d_head+(unsigned long)(ii*fifo->token_size);
    //printf("writing %p to output[%d]\n", d_pointer, ii);
    output[ii]=d_pointer;    
  }  

  fifo->head = d_head+i;
}

/* doen't actually use grgpu_fifo... */
void grgpu_fifo_pop_host(unsigned long *input, void*output, int length, int size) {
  //read through the input fifo
  int i=0;
  unsigned long prev = 0;
  unsigned long base = input[i];
  for(i=0; i<length; i++){
    if(input[i]<prev){
      base = input[i]-i*size;
      printf("Detected a wrap around at %p to %p. New base=%p\n", prev, input[i],base); 
      break;
    }
    prev = input[i];
  }
  printf("i=%d \n",i);
  cudaThreadSynchronize();
  cudaMemcpy(output, (void*)base, length*size, cudaMemcpyDeviceToHost);
  printf("quick check: %f\n", ((float*)output)[1023]);
  checkCUDAError("D2H: Out cudaMemcpy"); 
}

/* doesn't actually use grgpu_fifo... */
void * grgpu_fifo_pop_device(unsigned long *input, int length, int size) {
  //read through the input fifo
  int i=0;
  unsigned long prev = 0;
  unsigned long base = input[i];
  for(i=0; i<length; i++){
    if(input[i]<prev){
      base = input[i]-i*size;
      printf("Detected a wrap around at %p to %p. New base=%p\n", prev, input[i],base); 
      break;
    }
    prev = input[i];
  }
  return (void*)base;
}
