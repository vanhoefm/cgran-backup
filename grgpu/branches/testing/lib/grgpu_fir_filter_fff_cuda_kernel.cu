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

#define MAX_TAPS 60

// CHUNK_SIZE must be even and less than or equal TAPS.
#define MAX_CHUNK_SIZE 60

/* static int verbose=0; */
/* static float* d_taps=0; */
/* static float* h_taps=0; */
/* static int h_length=0; */
/* static int taps_changed=1; */
/* static int first_time=1; */
/* static float *d_odata; */

__global__ void fir_filter_fff_cuda_kernel(float* d_idata, float *d_odata, unsigned int elements_per_block, int noutput_items, float *d_taps, int ntaps, int chunk_size)
{
  // Every block will load n inputs and produce n outputs. 
  // Loading multiple inputs is done simultinously for every chunk of outputs.
  // The shared memory sdata will be implemented as  a circular buffer.
  
  // s_idata stores the inputs and s_multiplied_data stores the multiplication resutls.
  __shared__ float s_idata[MAX_CHUNK_SIZE];
  // As all threads are working in parallel, we need two arrays to store the partial resuls.
  // One to read from it and the other to store the new value. In every step, the role of the arrays changes.
  __shared__ float s_partial_results_i[MAX_TAPS];
  __shared__ float s_partial_results_j[MAX_TAPS];	
  __shared__ float s_final_results[MAX_CHUNK_SIZE];
    
    unsigned int tid = threadIdx.x;
	//printf("Kernel Launched in block:%d, thread %d .\n", blockIdx.x, tid);
	float tap_value = d_taps[tid];

	// First thread is responsible for initiating the partial sum and last thread 
	// writes the final result in a share array to store in the global memory later on

	unsigned int first_element, last_element, elements_to_process;
	first_element = blockIdx.x * elements_per_block;
	last_element = (first_element + elements_per_block > noutput_items) ? noutput_items : first_element + elements_per_block ;
	elements_to_process =  last_element - first_element;

/*
	printf("In block: %d, Thread:%d\n", blockIdx.x, threadIdx.x);
	printf("First Element:%d, Last Element:%d, Elements to process:%d\n", first_element, last_element, elements_to_process);
*/	

	//The input array d_idata is assumed to have the first delayed inputs first that we load in the next line.
	//The input_element_under_execution in the input array.
	unsigned int input_element_under_execution = 0; 
	unsigned int n_chunks = (elements_to_process + ntaps - 1) / chunk_size;
	unsigned int element_to_write_index = first_element; 
	//Number of produced output during the chunk execution.
	unsigned int outputs_to_write = 0;

	for (unsigned int i_chunk = 0; i_chunk < n_chunks; ++i_chunk) {
		// Some threads will load the input in parallel according to the following condition.
		if (tid < chunk_size) {
			s_idata[tid] = d_idata[first_element + tid +  i_chunk * chunk_size];
		}
		// Chunk execution.
		__syncthreads();
		outputs_to_write = 0;
		
		for (unsigned int i_chunk_element = 0; i_chunk_element < chunk_size; i_chunk_element += 2) {
			// Check if the thread is in a working phase.
			if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
				if (tid == 0) {
					s_partial_results_i[0] = s_idata[i_chunk_element] * tap_value;
				}
				else if (tid == ntaps - 1) {
					s_final_results[outputs_to_write] = s_partial_results_j[tid -1] + s_idata[i_chunk_element] * tap_value;
				}
				else {
					s_partial_results_i[tid] = 	s_partial_results_j[tid - 1] + s_idata[i_chunk_element] * tap_value;
				}
			}
			// The following condition tells every thread if he will contribute in writing the final results.
			if (input_element_under_execution >= ntaps -1) {
				++outputs_to_write;
			}
			++input_element_under_execution;
/*			printf("In phase i, input_element_under_execution: %d\n", input_element_under_execution - 1);
			printf("In block: %d, Thread:%d, i_chunk:%d, i_chunk_element:%d\n", blockIdx.x, threadIdx.x, i_chunk, i_chunk_element);
			printf("Outputs to write:%d\n", outputs_to_write);
			print_status(s_partial_results_i, s_partial_results_j, s_final_results, s_idata);  
			scanf("%d", &j);

 */			__syncthreads();	

			if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
				if (tid == 0) {
					s_partial_results_j[0] = s_idata[i_chunk_element + 1] * tap_value;
				}
				else if (tid == ntaps - 1) {
					s_final_results[outputs_to_write] = s_partial_results_i[tid -1] + s_idata[i_chunk_element + 1] * tap_value;
				}
				else {
					s_partial_results_j[tid] = 	s_partial_results_i[tid - 1] + s_idata[i_chunk_element + 1] * tap_value;
				}
			}
			if (input_element_under_execution >= ntaps -1) {
				++outputs_to_write;
			}
			++input_element_under_execution;
/*			printf("In phase j, input_element_under_execution: %d\n", input_element_under_execution - 1);
			printf("In block: %d, Thread:%d, i_chunk:%d, i_chunk_element:%d\n", blockIdx.x, threadIdx.x, i_chunk, i_chunk_element);
			printf("Outputs to write:%d\n", outputs_to_write);
			print_status(s_partial_results_i, s_partial_results_j, s_final_results, s_idata);  
			scanf("%d", &j);
*/
 			__syncthreads();
		} //End of chunk execution.
		// Load chunk resutls to global memory.
		if (tid < outputs_to_write) {
			d_odata[element_to_write_index + tid ] = s_final_results[tid];
/*			printf("In block: %d, Thread:%d, i_chunk:%d\n", blockIdx.x, threadIdx.x, i_chunk);
			printf("Element %d writen as value: %f\n", element_to_write_index + tid, s_final_results[tid]);
*/
		}//End of chunk execution.
		element_to_write_index += outputs_to_write;
	}// End of All chunks executions.
		

	// Execution of the same procedure for the last remaining elements. This will happen if the 
	// chunk size is not an integer multiple of elements_to_process.
	unsigned int remaining_elements = elements_to_process + ntaps -1 - n_chunks * chunk_size; 
	
	//printf("Remaining elements:%d\n", remaining_elements);
	if (tid < remaining_elements) {
		s_idata[tid] = d_idata[first_element + tid + n_chunks * chunk_size];
	}
	__syncthreads();
	outputs_to_write = 0;
	for (unsigned int i_chunk_element = 0; i_chunk_element < remaining_elements; i_chunk_element += 2) {
		// Check if the thread is in a working phase.
		if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
			if (tid == 0) {
				s_partial_results_i[0] = s_idata[i_chunk_element] * tap_value;
			}
			else if (tid == ntaps - 1) {
				s_final_results[outputs_to_write] = s_partial_results_j[tid -1] + s_idata[i_chunk_element] * tap_value;
			}
			else {
				s_partial_results_i[tid] = 	s_partial_results_j[tid - 1] + s_idata[i_chunk_element] * tap_value;
			}
		}
		if (input_element_under_execution >= ntaps -1) {
			++outputs_to_write;
		}
		++input_element_under_execution;
/*		printf("In phase i, input_element_under_execution: %d\n", input_element_under_execution - 1);
		printf("In block: %d, Thread:%d, i_chunk_element:%d\n", blockIdx.x, threadIdx.x, i_chunk_element);
		printf("Outputs to write:%d\n", outputs_to_write);
		print_status(s_partial_results_i, s_partial_results_j, s_final_results, s_idata);  
		scanf("%d", &j);
*/
 		__syncthreads();	
		// Execute this block if we need it only (the number of elements for this piece is even).	
		if (input_element_under_execution < elements_per_block + ntaps - 1) {
			if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
				if (tid == 0) {
					s_partial_results_j[0] = s_idata[i_chunk_element + 1] * tap_value;
				}
				else if (tid == ntaps - 1) {
					s_final_results[outputs_to_write] = s_partial_results_i[tid -1] + s_idata[i_chunk_element + 1] * tap_value;
				}
				else {
					s_partial_results_j[tid] = 	s_partial_results_i[tid - 1] + s_idata[i_chunk_element + 1] * tap_value;
				}
			}
			if (input_element_under_execution >= ntaps -1) {
				++outputs_to_write;
			}
		}
		++input_element_under_execution;
/*		printf("In phase j, input_element_under_execution: %d\n", input_element_under_execution - 1);
		printf("In block: %d, Thread:%d, i_chunk_element:%d\n", blockIdx.x, threadIdx.x, i_chunk_element);
		printf("Outputs to write:%d\n", outputs_to_write);
		print_status(s_partial_results_i, s_partial_results_j, s_final_results, s_idata);  
		scanf("%d", &j);
*/
 		__syncthreads();	
	} //End of chunk execution.
	// Load chunk resutls to global memory.
	if (tid < outputs_to_write) {
		d_odata[element_to_write_index + tid ] = s_final_results[tid];
/*		printf("In block: %d, Thread:%d\n", blockIdx.x, threadIdx.x);
		printf("Element %d writen as value: %f\n", element_to_write_index + tid, s_final_results[tid]);
*/
	}//End of chunk execution.

	//printf("Kernel End in block:%d, thread %d .\n", blockIdx.x, tid);
}



void grgpu_fir_filter_fff_cuda_work_device(int noutput_items, const unsigned long* input_items, unsigned long* output_items, void**d_taps, float*h_taps,int h_length, int verbose) 
{
  if(*d_taps==0x0){
    cudaMalloc(d_taps, /*i_size*/ 128*sizeof(float));
    checkCUDAError("Taps malloc");
    cudaMemcpy(*d_taps, h_taps, h_length * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("Memcpy taps");
  }

  // pointer for device memory
  float *d_idata = (float*)input_items[0];
  // WLP - in place editing of buffers
  float *d_odata = (float*)input_items[0];
  //unsigned int o_size = (noutput_items) * sizeof (float);
  //  unsigned int i_size = (h_length + noutput_items) * sizeof (float);

  //malloc the output array
  /* if(first_time){ */
  /*   first_time = 1; */
  /*   cudaMalloc( (void **) &d_odata, /\*o_size*\/70000*sizeof(float)); */
  /*   checkCUDAError("GRGPU FIR filter cudaMalloc"); */
  /* } */
  if(verbose) { 
    printf("FIR filter kernel starting. Inputs at %ld, Taps at %ld, %d outputs at %ld.\n", (long)d_idata, (long) *d_taps, noutput_items, (long)d_odata); 
  } 
	

  // The actual number of running blocks depends if N_OUTPUT is multiple of the required blocks.
  unsigned int running_blocks;
#define N_BLOCKS 80
  running_blocks = (noutput_items > N_BLOCKS) ? N_BLOCKS : noutput_items;
  unsigned int elements_per_block;
  elements_per_block = (noutput_items % running_blocks == 0)? noutput_items / running_blocks: noutput_items / running_blocks + 1;
  running_blocks = (noutput_items % elements_per_block == 0)? noutput_items/ elements_per_block:  noutput_items/ elements_per_block + 1;
  if(verbose) {
    printf("Elements per block:%d, Running blocks:%d\n", elements_per_block, running_blocks);
    printf("Got %d number of taps: %.2f, %.2f, %.2f ...\n", h_length, h_taps[0], h_taps[1], h_taps[2]);
  }
		
  dim3 dimGrid(running_blocks);
  dim3 dimBlock(h_length);

  //TODO: make sure chunk size is even
  fir_filter_fff_cuda_kernel<<<dimGrid, dimBlock>>>(d_idata, d_odata, elements_per_block, noutput_items, (float*)*d_taps, h_length, h_length);
  if(verbose) {
    printf("FIR filter kernel finished. %d outputs at %ld.\n", noutput_items, (long)d_odata);
  }

  checkCUDAError("kernel execution 1");

  cudaThreadSynchronize();
  //  cudaFree(d_idata);

  //now fill out the output output items array with the corresponding device pointers
  for(int i=0; i<1 /*noutput_items*/; i++){
    output_items[i]=(unsigned long)d_odata+i*8;
  }
}
