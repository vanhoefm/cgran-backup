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

#define TAPS 128

#define N_BLOCKS  80
// CHUNK_SIZE must be even and less than or equal TAPS.
#define CHUNK_SIZE 128

void cpuFunction(float *in_data, float *out_data, float* taps); 

/*__device__ int print_status(float * s_partial_i, float * s_partial_j, float * s_final, float * s_idata) {
	for (unsigned int i = 0; i < TAPS; ++i) {
		printf("s_partial_i[%d] = %f, s_partial_j[%d] = %f, TAPS[%d] = %f\n", i, s_partial_i[i], i, s_partial_j[i], i, d_taps[i]);
	}
	for (unsigned int i = 0; i < CHUNK_SIZE; ++i) {
		printf("s_idata[%d] = %f, s_final[%d] = %f\n", i, s_idata[i], i, s_final[i]);
	}
	return 0;
}*/
// Part 3 of 5: implement the kernel
__global__ void fir_fff_cuda_kernel(float* d_idata, float *d_odata, unsigned int elements_per_block, int noutput_items, float *d_taps)
{
  // Every block will load n inputs and produce n outputs. 
  // Loading multiple inputs is done simultinously for every chunk of outputs.
  // The shared memory sdata will be implemented as  a circular buffer.
  
  // s_idata stores the inputs and s_multiplied_data stores the multiplication resutls.
  __shared__ float s_idata[CHUNK_SIZE];
  // As all threads are working in parallel, we need two arrays to store the partial resuls.
  // One to read from it and the other to store the new value. In every step, the role of the arrays changes.
  __shared__ float s_partial_results_i[TAPS];
  __shared__ float s_partial_results_j[TAPS];	
    __shared__ float s_final_results[CHUNK_SIZE];

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
	unsigned int n_chunks = (elements_to_process + TAPS - 1) / CHUNK_SIZE;
	unsigned int element_to_write_index = first_element; 
	//Number of produced output during the chunk execution.
	unsigned int outputs_to_write = 0;

//	unsigned int j;


/*
	printf("n_chunks:%d\n", n_chunks);
	scanf("%d", &j);
*/
	for (unsigned int i_chunk = 0; i_chunk < n_chunks; ++i_chunk) {
		// Some threads will load the input in parallel according to the following condition.
		if (tid < CHUNK_SIZE) {
			s_idata[tid] = d_idata[first_element + tid +  i_chunk * CHUNK_SIZE];
		}
		// Chunk execution.
		__syncthreads();
		outputs_to_write = 0;
		
		for (unsigned int i_chunk_element = 0; i_chunk_element < CHUNK_SIZE; i_chunk_element += 2) {
			// Check if the thread is in a working phase.
			if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
				if (tid == 0) {
					s_partial_results_i[0] = s_idata[i_chunk_element] * tap_value;
				}
				else if (tid == TAPS - 1) {
					s_final_results[outputs_to_write] = s_partial_results_j[tid -1] + s_idata[i_chunk_element] * tap_value;
				}
				else {
					s_partial_results_i[tid] = 	s_partial_results_j[tid - 1] + s_idata[i_chunk_element] * tap_value;
				}
			}
			// The following condition tells every thread if he will contribute in writing the final results.
			if (input_element_under_execution >= TAPS -1) {
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
				else if (tid == TAPS - 1) {
					s_final_results[outputs_to_write] = s_partial_results_i[tid -1] + s_idata[i_chunk_element + 1] * tap_value;
				}
				else {
					s_partial_results_j[tid] = 	s_partial_results_i[tid - 1] + s_idata[i_chunk_element + 1] * tap_value;
				}
			}
			if (input_element_under_execution >= TAPS -1) {
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
	unsigned int remaining_elements = elements_to_process + TAPS -1 - n_chunks * CHUNK_SIZE; 
	
	//printf("Remaining elements:%d\n", remaining_elements);
	if (tid < remaining_elements) {
		s_idata[tid] = d_idata[first_element + tid + n_chunks * CHUNK_SIZE];
	}
	__syncthreads();
	outputs_to_write = 0;
	for (unsigned int i_chunk_element = 0; i_chunk_element < remaining_elements; i_chunk_element += 2) {
		// Check if the thread is in a working phase.
		if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
			if (tid == 0) {
				s_partial_results_i[0] = s_idata[i_chunk_element] * tap_value;
			}
			else if (tid == TAPS - 1) {
				s_final_results[outputs_to_write] = s_partial_results_j[tid -1] + s_idata[i_chunk_element] * tap_value;
			}
			else {
				s_partial_results_i[tid] = 	s_partial_results_j[tid - 1] + s_idata[i_chunk_element] * tap_value;
			}
		}
		if (input_element_under_execution >= TAPS -1) {
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
		if (input_element_under_execution < elements_per_block + TAPS - 1) {
			if ( (input_element_under_execution >= tid) && (input_element_under_execution < elements_to_process + tid)) {
				if (tid == 0) {
					s_partial_results_j[0] = s_idata[i_chunk_element + 1] * tap_value;
				}
				else if (tid == TAPS - 1) {
					s_final_results[outputs_to_write] = s_partial_results_i[tid -1] + s_idata[i_chunk_element + 1] * tap_value;
				}
				else {
					s_partial_results_j[tid] = 	s_partial_results_i[tid - 1] + s_idata[i_chunk_element + 1] * tap_value;
				}
			}
			if (input_element_under_execution >= TAPS -1) {
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

//extern "C"
void grgpu_fir_fff_cuda_work_device(int noutput_items, const float* input_items,float* output_items) 
{

  // The actual number of running blocks depends if N_OUTPUT is multiple of the required blocks.
  unsigned int running_blocks;
  running_blocks = (noutput_items > N_BLOCKS) ? N_BLOCKS : noutput_items;
  unsigned int elements_per_block;
  elements_per_block = (noutput_items % running_blocks == 0)? noutput_items / running_blocks: noutput_items / running_blocks + 1;
  running_blocks = (noutput_items % elements_per_block == 0)? noutput_items/ elements_per_block:  noutput_items/ elements_per_block + 1;
  //unsigned int running_timer, total_timer, host_timer;
  //cutCreateTimer(&running_timer);
  //cutCreateTimer(&total_timer);
  //cutCreateTimer(&host_timer);
  //double gpuTime;
  //double totalTime;
  

  // pointer for device memory
  float *d_idata, *d_odata;
  unsigned int i_size = (TAPS + noutput_items) * sizeof (float);
  unsigned int o_size = (noutput_items) * sizeof (float);
  //Initialze taps.

  float h_taps[TAPS], *d_taps;
  
  for (int i = 0; i < TAPS; ++i) {
    h_taps[i] = 0.0;
  }
  h_taps[0] = 0.5;
  h_taps[1] = 0.5;

  printf("Elements per block:%d, Running blocks:%d\n", elements_per_block, running_blocks);
	
  //cutResetTimer(total_timer);
  cudaMalloc( (void **) &d_idata, i_size);
  cudaMalloc( (void **) &d_odata, o_size);
  cudaMalloc( (void **) &d_taps, TAPS * sizeof(float));

  checkCUDAError("Malloc");
		
  //cutStartTimer(total_timer);
  cudaMemcpy(d_taps, h_taps, TAPS * sizeof(float), cudaMemcpyHostToDevice); 	
  checkCUDAError("Memcpy taps");
  cudaMemcpy(d_idata, input_items, i_size, cudaMemcpyHostToDevice);
  checkCUDAError("Memcpy idata");
  
  //cutResetTimer(running_timer);
  
  // Part 2 of 5: configure and launch kernel
    
  dim3 dimGrid(running_blocks);
  dim3 dimBlock(TAPS);

  cudaThreadSynchronize();
  //cutStartTimer(running_timer);
  fir_fff_cuda_kernel<<<dimGrid, dimBlock>>>(d_idata, d_odata, elements_per_block, noutput_items, d_taps);

  // block until the device has completed
  cudaThreadSynchronize();

  // check if kernel execution generated an error. disabled 
  // for time computation.
  checkCUDAError("kernel execution 1");
	
  // Part 4 of 5: device to host copy
  //cutStopTimer(running_timer);
  cudaMemcpy(output_items, d_odata, o_size, cudaMemcpyDeviceToHost);

  //cutStopTimer(total_timer);
  // Check for any CUDA errors
    checkCUDAError("Out cudaMemcpy");

   
    // free device memory
    //    cudaFree(d_idata);
    cudaFree(d_odata); 

	//gpuTime = cutGetTimerValue(running_timer);
	//totalTime = cutGetTimerValue(total_timer);
	//printf("\n Device Time: %f msec\n", gpuTime); 
	//printf("\n Total Time: %f msec\n", totalTime); 


}
