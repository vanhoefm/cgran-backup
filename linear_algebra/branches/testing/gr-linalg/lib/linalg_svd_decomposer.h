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
#ifndef INCLUDED_LINALG_SVD_DECOMPOSER_FF_H
#define INCLUDED_LINALG_SVD_DECOMPOSER_FF_H

#include <boost/shared_ptr.hpp>
#include <gr_complex.h>

extern "C"
{
  void
  zgesdd_ (char* JOBZ, size_t* M, size_t* N, std::complex<double>* A,
           size_t* LDA, double* S, std::complex<double>* U, size_t* LDU,
           std::complex<double>* VT, size_t* LDVT, std::complex<double>* WORK,
           size_t* LWORK, double* RWORK, int* IWORK, int* INFO);
  void
  zgesvd_ (char* JOBU, char* JOBVT, size_t* M, size_t* N, std::complex<double>* A,
           size_t* LDA, double* S, std::complex<double>* U, size_t* LDU,
           std::complex<double>* VT, size_t* LDVT, std::complex<double>* WORK,
           size_t* LWORK, double* RWORK, int* INFO);
}

class linalg_svd_decomposer;

typedef boost::shared_ptr<linalg_svd_decomposer> linalg_svd_decomposer_sptr;

/*!
 * \brief Return a shared_ptr to a new instance of linalg_svd_decomposer.
 */
linalg_svd_decomposer_sptr linalg_make_svd_decomposer (size_t M, size_t N);

/*!
 * \brief Decomposes Matrix A into A = U*S*V^T based on LAPACK's zgsevdd
 * \ingroup linalg
 */
class linalg_svd_decomposer
{
private:
  
  friend linalg_svd_decomposer_sptr linalg_make_svd_decomposer (size_t M, size_t N);

  size_t d_M, d_N, d_LWORK, d_LDVT, d_LDU, d_LDA;
  gr_complexd* d_mat_A;
  gr_complexd* d_mat_U;
  gr_complexd* d_mat_VT;
  gr_complexd* d_WORK;
  double* d_RWORK;
  double* d_S;
  int* d_IWORK;
  int d_INFO;
  linalg_svd_decomposer (size_t M, size_t N);

 public:
  ~linalg_svd_decomposer ();
  double* S ();
  gr_complexd* U ();
  gr_complexd* VT ();
  void set_matrix (gr_complexd* mat_A);
  void decompose ();

};

class linalg_svd_decomposer_no_vt;
typedef boost::shared_ptr<linalg_svd_decomposer_no_vt> linalg_svd_decomposer_no_vt_sptr;


linalg_svd_decomposer_no_vt_sptr linalg_make_svd_decomposer_no_vt (size_t M, size_t N);

/*!
* \brief Decomposes Matrix A into U*S*V^T but V is not calculated, based on LAPACK's zgesvd
* \ingroup linalg
*/
class linalg_svd_decomposer_no_vt
{
private:
  friend linalg_svd_decomposer_no_vt_sptr 
         linalg_make_svd_decomposer_no_vt (size_t M, size_t N); 

  size_t d_M, d_N, d_LDA, d_LDU, d_LDVT, d_LWORK;
  gr_complexd* d_mat_A;
  gr_complexd* d_mat_U;
  gr_complexd* d_mat_VT;
  gr_complexd* d_WORK;
  double* d_RWORK;
  double* d_S;
  int d_INFO;
  linalg_svd_decomposer_no_vt (size_t M, size_t N);

public:
  ~linalg_svd_decomposer_no_vt ();
  double* S ();
  gr_complexd* U();
  void set_matrix (gr_complexd* mat_A);
  void decompose ();
  
};
#endif /* INCLUDED_LINALG_SVD_DECOMPOSER_FF_H */
