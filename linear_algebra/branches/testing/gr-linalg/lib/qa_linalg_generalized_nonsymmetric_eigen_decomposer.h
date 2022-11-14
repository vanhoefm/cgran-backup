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
#ifndef INCLUDED_QA_LINALG_GENERALIZED_NONSYMMETRIC_EIGEN_DECOMPOSER_H
#define INCLUDED_QA_LINALG_GENERALIZED_NONSYMMETRIC_EIGEN_DECOMPOSER_H

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>

class qa_linalg_generalized_nonsymmetric_eigen_decomposer : public CppUnit::TestCase {

  CPPUNIT_TEST_SUITE (qa_linalg_generalized_nonsymmetric_eigen_decomposer);
  CPPUNIT_TEST (test_hilbert);
  CPPUNIT_TEST (test_hilbert_magic);
  CPPUNIT_TEST_SUITE_END ();

 private:
  void test_hilbert ();
  void test_hilbert_magic ();
};

#endif /* INCLUDED_QA_LINALG_GENERALIZED_NONSYMMETRIC_EIGEN_DECOMPOSER_H */
