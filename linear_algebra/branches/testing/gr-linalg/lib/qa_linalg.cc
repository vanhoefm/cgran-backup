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

/*
 * This class gathers together all the test cases for the example
 * directory into a single test suite.  As you create new test cases,
 * add them here.
 */

#include <qa_linalg.h>
#include <qa_linalg_svd_decomposer.h>
#include <qa_linalg_generalized_nonsymmetric_eigen_decomposer.h>

CppUnit::TestSuite *
qa_linalg::suite()
{
  CppUnit::TestSuite *s = new CppUnit::TestSuite("linalg");

  s->addTest(qa_linalg_svd_decomposer::suite());
  s->addTest(qa_linalg_generalized_nonsymmetric_eigen_decomposer::suite());
  return s;
}
