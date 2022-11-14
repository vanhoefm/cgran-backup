/*
 * Copyright 2010 Karlsruhe Institute of Technology, Communications Engineering Lab
 * 
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

#include <gr_complex.h>
#include <cstdio>

void
linalg_helpers_print_matrix (gr_complexd* matrix, size_t M, size_t N, const char* title)
{
  printf("\nMatrix \"%s\" %s:\n\n", title,"(transposed)");
  for(int m = 0; m < M; m++)
  {
    for(int n = 0; n < N; n++)
    {
      printf("(%3f,%3f)\t",matrix[m*N+n].real(), matrix[m*N+n].imag());
    }
    printf("\n");
  }
}
