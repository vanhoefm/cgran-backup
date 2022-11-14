/* -*- c++ -*- */
/*
 * Copyright 2009 Free Software Foundation, Inc.
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
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <qa_linalg_svd_decomposer.h>
#include <linalg_svd_decomposer.h>
#include <cppunit/TestAssert.h>

/*
* Set-up with positive semidefinite system (hilb(3) in octave)
*/

void
qa_linalg_svd_decomposer::test_hilbert()
{
  linalg_svd_decomposer_sptr svd = linalg_make_svd_decomposer (3,4); 
  std::vector<gr_complexd> matrix;

  /* setup col 0 */
  matrix.push_back (gr_complexd(1,0));
  matrix.push_back (gr_complexd(1.0/2,0));
  matrix.push_back (gr_complexd(1.0/3,0));

  /* setup col 1 */ 
  matrix.push_back (gr_complexd(1.0/2,0));
  matrix.push_back (gr_complexd(1.0/3,0));
  matrix.push_back (gr_complexd(1.0/4,0));

  /* setup col 2 */ 
  matrix.push_back (gr_complexd(1.0/3,0));
  matrix.push_back (gr_complexd(1.0/4,0));
  matrix.push_back (gr_complexd(1.0/5,0));
  
  /* setup col 3 */
  matrix.push_back (gr_complexd(1.0/4,0));
  matrix.push_back (gr_complexd(1.0/5,0));
  matrix.push_back (gr_complexd(1.0/6,0));


  CPPUNIT_ASSERT_EQUAL ((size_t) 12, matrix.size());
  svd->set_matrix(&matrix[0]);
  svd->decompose();
  double* singular_values = new double[3];
  singular_values = svd->S(); 

  /* TODO howto ASSERT_ALMOST_EQUAL? 
     compare with singular values calculated by GNU octave */
  CPPUNIT_ASSERT (singular_values[0] - 1.4519142 < 0.000001);
  CPPUNIT_ASSERT (singular_values[1] - 0.1433123 < 0.000001);
  CPPUNIT_ASSERT (singular_values[2] - 0.0042289 < 0.000001);
  
}

void
qa_linalg_svd_decomposer::test_hilbert2()
{
  linalg_svd_decomposer_no_vt_sptr svd = linalg_make_svd_decomposer_no_vt (3,3); 
  std::vector<gr_complexd> matrix;

  /* setup col 0 */
  matrix.push_back (gr_complexd(1,0));
  matrix.push_back (gr_complexd(1.0/2,0));
  matrix.push_back (gr_complexd(1.0/3,0));

  /* setup col 1 */ 
  matrix.push_back (gr_complexd(1.0/2,0));
  matrix.push_back (gr_complexd(1.0/3,0));
  matrix.push_back (gr_complexd(1.0/4,0));

  /* setup col 2 */ 
  matrix.push_back (gr_complexd(1.0/3,0));
  matrix.push_back (gr_complexd(1.0/4,0));
  matrix.push_back (gr_complexd(1.0/5,0));
  
  CPPUNIT_ASSERT_EQUAL ((size_t) 9, matrix.size());
  svd->set_matrix(&matrix[0]);
  svd->decompose();
  double* singular_values = new double[3];
  singular_values = svd->S(); 

  /* TODO howto ASSERT_ALMOST_EQUAL? 
     compare with singular values calculated by GNU octave */
  CPPUNIT_ASSERT (singular_values[0] - 1.4083189 < 0.000001);
  CPPUNIT_ASSERT (singular_values[1] - 0.1223271 < 0.000001);
  CPPUNIT_ASSERT (singular_values[2] - 0.0026873 < 0.000001);
  
}
