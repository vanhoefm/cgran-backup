#ifndef INCLUDED_GRGPU_ADD_CONST_FF_CUDA_KERNEL_H
#define INCLUDED_GRGPU_ADD_CONST_FF_CUDA_KERNEL_H
#include "grgpu_utils.h"

typedef struct {
  float constant;
  grgpu_fifo *fifo_context;
} grgpu_add_const_ff_cuda_context;

#endif
