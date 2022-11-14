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
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <qa_linalg_generalized_nonsymmetric_eigen_decomposer.h>
#include <linalg_generalized_nonsymmetric_eigen_decomposer.h>
#include <linalg_helpers.h>
#include <cppunit/TestAssert.h>

/*
* Set-up with positive semidefinite system (hilb(3) in Octave)
*/

void
qa_linalg_generalized_nonsymmetric_eigen_decomposer::test_hilbert()
{
  linalg_generalized_nonsymmetric_eigen_decomposer_sptr evd = linalg_make_generalized_nonsymmetric_eigen_decomposer (3, false, false); 
  std::vector<gr_complexd> A, B;

  /* setup col 0 */
  A.push_back (gr_complexd(1,0));
  A.push_back (gr_complexd(1.0/2,0));
  A.push_back (gr_complexd(1.0/3,0));

  /* setup col 1 */ 
  A.push_back (gr_complexd(1.0/2,0));
  A.push_back (gr_complexd(1.0/3,0));
  A.push_back (gr_complexd(1.0/4,0));

  /* setup col 2 */ 
  A.push_back (gr_complexd(1.0/3,0));
  A.push_back (gr_complexd(1.0/4,0));
  A.push_back (gr_complexd(1.0/5,0));

  /* setup B col 0 */
  B.push_back (gr_complexd(1.0,0));
  B.push_back (gr_complexd(0,0));
  B.push_back (gr_complexd(0,0));

  /* setup B col 1 */
  B.push_back (gr_complexd(0,0));
  B.push_back (gr_complexd(1.0,0));
  B.push_back (gr_complexd(0,0));

  /* setup B col 2 */
  B.push_back (gr_complexd(0,0));
  B.push_back (gr_complexd(0,0));
  B.push_back (gr_complexd(1.0,0));

  evd->set_matrices(&A[0], &B[0]);
  evd->decompose();
  gr_complexd* ALPHA; 
  gr_complexd* BETA;
  ALPHA = evd->ALPHA (); 
  BETA = evd->BETA ();

  /* TODO howto ASSERT_ALMOST_EQUAL? 
     compare with eigenvalues calculated by GNU Octave */
  CPPUNIT_ASSERT (std::abs(ALPHA[2]/BETA[2]) - 0.0026873 < 0.000001);
  CPPUNIT_ASSERT (std::abs(ALPHA[1]/BETA[1]) - 0.1223271 < 0.000001);
  CPPUNIT_ASSERT (std::abs(ALPHA[0]/BETA[0]) - 1.4083189 < 0.000001);
}

void
qa_linalg_generalized_nonsymmetric_eigen_decomposer::test_hilbert_magic()
{
  linalg_generalized_nonsymmetric_eigen_decomposer_sptr evd = linalg_make_generalized_nonsymmetric_eigen_decomposer (3, true, true); 
  std::vector<gr_complexd> A, B;

  /* setup col 0 */
  A.push_back (gr_complexd(1,0));
  A.push_back (gr_complexd(1.0/2,0));
  A.push_back (gr_complexd(1.0/3,0));

  /* setup col 1 */ 
  A.push_back (gr_complexd(1.0/2,0));
  A.push_back (gr_complexd(1.0/3,0));
  A.push_back (gr_complexd(1.0/4,0));

  /* setup col 2 */ 
  A.push_back (gr_complexd(1.0/3,0));
  A.push_back (gr_complexd(1.0/4,0));
  A.push_back (gr_complexd(1.0/5,0));

  /* setup B col 0 */
  B.push_back (gr_complexd(8.0,0));
  B.push_back (gr_complexd(3.0,0));
  B.push_back (gr_complexd(4.0,0));

  /* setup B col 1 */
  B.push_back (gr_complexd(1.0,0));
  B.push_back (gr_complexd(5.0,0));
  B.push_back (gr_complexd(9.0,0));

  /* setup B col 2 */
  B.push_back (gr_complexd(6.0,0));
  B.push_back (gr_complexd(7.0,0));
  B.push_back (gr_complexd(2.0,0));

  evd->set_matrices(&A[0], &B[0]);
  evd->decompose();
  gr_complexd* ALPHA; 
  gr_complexd* BETA;
  ALPHA = evd->ALPHA (); 
  BETA = evd->BETA ();

  /* TODO howto ASSERT_ALMOST_EQUAL? 
     compare with eigenvalues calculated by GNU Octave */
  CPPUNIT_ASSERT (std::abs (ALPHA[2]/BETA[2]) - 0.10296 < 0.000001);
  CPPUNIT_ASSERT (std::abs (ALPHA[1]/BETA[1]) - 0.00071482 < 0.000001);
  CPPUNIT_ASSERT (std::abs (ALPHA[0]/BETA[0]) - 0.17473 < 0.000001);

  gr_complexd* right_eigenvectors;
  right_eigenvectors = evd-> VR();

  /* compare right eigenvectors  with the ones calculated by GNU Octave */
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[0]) -  1.0 < 0.000001);
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[1]) - 0.120006  < 0.000001);
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[2]) -  0.45223 < 0.000001);
  
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[3]) -  0.159454 < 0.000001);
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[4]) - 0.994554  < 0.000001);

  CPPUNIT_ASSERT (std::abs (right_eigenvectors[5]) -  1.0 < 0.000001);

  CPPUNIT_ASSERT (std::abs (right_eigenvectors[6]) -  0.714672 < 0.000001);
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[7]) - 1.0  < 0.000001);
  CPPUNIT_ASSERT (std::abs (right_eigenvectors[8]) -  0.578769 < 0.000001);
}
