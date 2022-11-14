#
# Copyright 2004,2005,2006,2008,2009,2010 Free Software Foundation, Inc.
# 
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

include $(top_srcdir)/Makefile.common

# list of programs run by "make check" and "make distcheck"
TESTS = test_all

# ----------------------------------------------------------------
# linalg C++ library: libgnuradio-linalg.so
# ----------------------------------------------------------------

# C/C++ headers get installed in ${prefix}/include/$(modname)
modinclude_HEADERS = \
	linalg_svd_decomposer.h \
	linalg_generalized_nonsymmetric_eigen_decomposer.h

lib_LTLIBRARIES = libgnuradio-linalg.la

libgnuradio_linalg_la_SOURCES = \
	linalg_svd_decomposer.cc \
        linalg_generalized_nonsymmetric_eigen_decomposer.cc

libgnuradio_linalg_la_LIBADD = \
	$(GNURADIO_CORE_LA)

libgnuradio_linalg_la_LDFLAGS = \
	$(NO_UNDEFINED) \
        -latlas \
        -llapack \
        -lblas \
        -lm

# ----------------------------------------------------------------
# linalg C++ QA library: libgnuradio-linalg-qa.so (not installed)
# ----------------------------------------------------------------

noinst_LTLIBRARIES = libgnuradio-linalg-qa.la

libgnuradio_linalg_qa_la_SOURCES = \
	qa_linalg.cc \
	qa_linalg_svd_decomposer.cc \
	qa_linalg_generalized_nonsymmetric_eigen_decomposer.cc \
        linalg_helpers.cc

libgnuradio_linalg_qa_la_LDFLAGS = $(NO_UNDEFINED)\
                                   -version-info 0:0:0 \
				   -latlas\
                                   -llapack\
                                   -lblas\
                                   -lm

libgnuradio_linalg_qa_la_LIBADD = \
	libgnuradio-linalg.la \
	$(CPPUNIT_LIBS)

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
noinst_HEADERS = \
	qa_linalg.h \
	qa_linalg_svd_decomposer.h \
	qa_linalg_generalized_nonsymmetric_eigen_decomposer.h \
        linalg_helpers.h

# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS = \
	test_all

test_all_SOURCES = test_all.cc
test_all_LDADD   = libgnuradio-linalg-qa.la
