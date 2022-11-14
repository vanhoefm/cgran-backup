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
 */

/*
 * This class gathers together all the test cases for the example
 * directory into a single test suite.  As you create new test cases,
 * add them here.
 */

#include <qa_grgpu.h>
#include <qa_grgpu_fir_fff_cuda.h>
//#include <qa_grgpu_square2_ff.h>

CppUnit::TestSuite *
qa_grgpu::suite()
{
  CppUnit::TestSuite *s = new CppUnit::TestSuite("grgpu");

  s->addTest(qa_grgpu_fir_fff_cuda::suite());
  //  s->addTest(qa_grgpu_square2_ff::suite());

  return s;
}
