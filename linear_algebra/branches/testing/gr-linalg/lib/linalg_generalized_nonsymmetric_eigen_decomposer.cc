/* -*- c++ -*- */
/*
 * Copyright 2010 Karlsruhe Institute of Technology, Communications Engineering Lab
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

/*
 * config.h is generated by configure.  It contains the results
 * of probing for features, options etc.  It should be the first
 * file included in your .cc file.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <linalg_generalized_nonsymmetric_eigen_decomposer.h>
#include <algorithm>
#include <cstdio> //TODO remove

/*
 * Create a new instance of linalg_generalized_nonsymmetric_eigen_decomposer and return
 * a boost shared_ptr.  This is effectively the public constructor.
 */
linalg_generalized_nonsymmetric_eigen_decomposer_sptr 
linalg_make_generalized_nonsymmetric_eigen_decomposer (size_t N, bool JOBVL, bool JOBVR)
{
  return linalg_generalized_nonsymmetric_eigen_decomposer_sptr (new linalg_generalized_nonsymmetric_eigen_decomposer (N, JOBVL, JOBVR));
}

linalg_generalized_nonsymmetric_eigen_decomposer::linalg_generalized_nonsymmetric_eigen_decomposer (size_t N, bool JOBVL, bool JOBVR) : d_N(N), d_JOBVL((JOBVL)? 'V':'N'), d_JOBVR((JOBVR)? 'V':'N')
{
  d_LDA = std::max (static_cast<size_t> (1), N);
  d_LDB = d_LDA;  // it's the same calculation as above
  d_ALPHA = new gr_complexd[N];
  d_BETA = new gr_complexd[N]; 

  /* if we do want left GEVectors, we need space for them */
  if (JOBVL)
  {
    d_LDVL = N;
    d_mat_VL = new gr_complexd[d_LDVL*N];
    printf("[GNEVD] Allocated %d bytes for calculation of left eigenvectors\n", d_LDVL * sizeof(gr_complexd));

  }
  else
  {
    d_LDVL = 1;
    d_mat_VL = NULL;
  }

  /* if we do want right GEVectors, we need space for them */
  if (JOBVR)
  { 
    d_LDVR = N;
    d_mat_VR = new gr_complexd[d_LDVR*N];
    printf("[GNEVD] Allocated %d bytes for calculation of right eigenvectors\n", d_LDVR * sizeof(gr_complexd));
  }
  else 
  {
    d_LDVR = 1;
    d_mat_VR = NULL;
  }
  d_RWORK = new double[8*N];
  
  /* determine optimal workspace size */
  int tmpinfo = 0;
  gr_complexd tmpwork;
  size_t tmplwork = -1;
  zggev_ (&d_JOBVL, &d_JOBVR, &d_N, d_mat_A, &d_LDA, d_mat_B, &d_LDB, d_ALPHA,
          d_BETA, d_mat_VL, &d_LDVL, d_mat_VR, &d_LDVR, &tmpwork, &tmplwork, d_RWORK,
          &tmpinfo);
  
  
  /* use LAPACK's parameter checking before allocating mem */
  if (!tmpinfo)
  {
    d_LWORK = static_cast<size_t> (std::ceil (tmpwork.real ()));
    d_WORK = new gr_complexd[d_LWORK];
    printf("[GNEVD] Allocated %d bytes for WORK\n", d_LWORK * sizeof(gr_complexd));
  }
  else
  {
    throw "Invalid LAPACK parameters!";
    d_WORK = NULL;
  }
  
}

linalg_generalized_nonsymmetric_eigen_decomposer::~linalg_generalized_nonsymmetric_eigen_decomposer ()
{
  delete[] d_ALPHA;
  delete[] d_BETA;
  if (d_mat_VL != NULL) delete[] d_mat_VL;
  if (d_mat_VR != NULL) delete[] d_mat_VR;
  delete[] d_RWORK;
  if (d_LWORK != NULL) delete[] d_WORK;
}

void
linalg_generalized_nonsymmetric_eigen_decomposer::set_matrices (gr_complexd* mat_A, gr_complexd* mat_B)
{
  d_mat_A = mat_A;
  d_mat_B = mat_B;
}

void
linalg_generalized_nonsymmetric_eigen_decomposer::decompose ()
{
  zggev_ (&d_JOBVL, &d_JOBVR, &d_N, d_mat_A, &d_LDA, d_mat_B, &d_LDB, d_ALPHA,
          d_BETA, d_mat_VL, &d_LDVL, d_mat_VR, &d_LDVR, d_WORK, &d_LWORK, d_RWORK,
          &d_INFO);
  //printf("\n\nINFO: %d", d_INFO);
}

gr_complexd*
linalg_generalized_nonsymmetric_eigen_decomposer::ALPHA ()
{
  return d_ALPHA;
}

gr_complexd*
linalg_generalized_nonsymmetric_eigen_decomposer::BETA ()
{
  return d_BETA;
}

gr_complexd*
linalg_generalized_nonsymmetric_eigen_decomposer::VL ()
{
  return d_mat_VL;
}

gr_complexd*
linalg_generalized_nonsymmetric_eigen_decomposer::VR ()
{
  return d_mat_VR;
}