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


#ifndef INCLUDED_GRGPU_UTILS_H
#define INCLUDED_GRGPU_UTILS_H

typedef struct {
  unsigned long head;
  int length;
  int history;
  int multiple;
  int wrap_pad;
  int token_size;
  unsigned long buffer;
} grgpu_fifo;

void checkCUDAError(const char *msg);

void grgpu_fifo_push(grgpu_fifo *fifo, void*input, unsigned long *output, int length);
void grgpu_fifo_pop_host(unsigned long *input, void*output, int length,  int size);
void * grgpu_fifo_pop_device(unsigned long *input, int length, int size);
#endif

