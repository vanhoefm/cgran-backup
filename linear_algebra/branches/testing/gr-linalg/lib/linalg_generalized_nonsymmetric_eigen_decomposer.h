/* -*- c++ -*- */
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
#ifndef INCLUDED_LINALG_GENERALIZED_NONSYMMETRIC_EIGEN_DECOMPOSER_FF_H
#define INCLUDED_LINALG_GENERALIZED_NONSYMMETRIC_EIGEN_DECOMPOSER_FF_H

#include <boost/shared_ptr.hpp>
#include <gr_complex.h>

extern "C"
{
  void
  zggev_ (char* JOBVL, char* JOBVR, size_t* N, std::complex<double>* A, size_t* LDA,
          std::complex<double>* B, size_t* LDB, std::complex<double>* ALPHA,
	  std::complex<double>* BETA, std::complex<double>* VL, size_t* LDVL, 
          std::complex<double>* VR, size_t* LDVR, std::complex<double>* WORK,
          size_t* LWORK, double* RWORK, int* INFO);          
}

class linalg_generalized_nonsymmetric_eigen_decomposer;

typedef boost::shared_ptr<linalg_generalized_nonsymmetric_eigen_decomposer> 
linalg_generalized_nonsymmetric_eigen_decomposer_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of linalg_generalized_nonsymmetric_eigen_decomposer.
 */
linalg_generalized_nonsymmetric_eigen_decomposer_sptr 
linalg_make_generalized_nonsymmetric_eigen_decomposer (size_t N, bool JOBVL, bool JOBVR);

/*!
 * \brief Decomposes Matrix A into A = U*S*V^T based on LAPACK's zgsevdd
 * \ingroup linalg
 */
class linalg_generalized_nonsymmetric_eigen_decomposer
{
private:
  
  friend linalg_generalized_nonsymmetric_eigen_decomposer_sptr 
  linalg_make_generalized_nonsymmetric_eigen_decomposer (size_t N, bool JOBVL, bool JOBVR);

  /* Bounds of the Matrices */
  size_t d_N, d_LDA, d_LDB, d_LDVL, d_LDVR, d_LWORK;
  char d_JOBVL, d_JOBVR;

  /* Matrix storage used */
  gr_complexd* d_mat_A;
  gr_complexd* d_mat_B;
  gr_complexd* d_mat_VL; // if d_JOBVL is true, this will hold the left GEVectors
  gr_complexd* d_mat_VR; // if d_JOBVR is true, this will hold the right GEVectors
  gr_complexd* d_ALPHA;
  gr_complexd* d_BETA;
  gr_complexd* d_WORK;

  double* d_RWORK;
  int d_INFO;
  linalg_generalized_nonsymmetric_eigen_decomposer (size_t N, bool JOBVL, bool JOBVR);

 public:
  ~linalg_generalized_nonsymmetric_eigen_decomposer ();
  double* S ();
  gr_complexd* VL (); 
  gr_complexd* VR ();

  /* after solving alpha(k) / beta(k) will be the generalized eigenvectors */
  gr_complexd* ALPHA (); 
  gr_complexd* BETA ();
  void set_matrices (gr_complexd* mat_A, gr_complexd* mat_B);
  void decompose ();

};
#endif /* INCLUDED_LINALG_GENERALIZED_NONSYMMETRIC_EIGEN_DECOMPOSER_FF_H */
