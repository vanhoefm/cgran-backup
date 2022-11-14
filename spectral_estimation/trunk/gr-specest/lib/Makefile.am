#
# Copyright 2004,2005,2006,2008,2009,2010 Free Software Foundation, Inc.
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
# specest C++ library: libgnuradio-specest.so
# ----------------------------------------------------------------

# C/C++ headers get installed in ${prefix}/include/$(modname)
modinclude_HEADERS = \
	specesti_arburg.h 		\
	specesti_arfcov.h		\
	specesti_arfmcov.h		\
	specest_pad_vector.h 		\
	specest_adaptiveweighting_vff.h	\
	specest_moving_average_vff.h 	\
	specest_reciprocal_ff.h 	\
	specest_arburg_vcc.h 		\
	specest_burg.h 			\
	specest_arfcov_vcc.h		\
	specest_fcov.h			\
	specest_arfmcov_vcc.h		\
	specest_fmcov.h			\
	specest_welch.h 		\
	specest_stream_to_vector_overlap.h

lib_LTLIBRARIES = libgnuradio-specest.la

libgnuradio_specest_la_SOURCES = \
	specesti_arburg.cc 			\
	specesti_arfcov.cc			\
	specesti_arfmcov.cc			\
	specest_arburg_vcc.cc 			\
	specest_arfcov_vcc.cc			\
	specest_arfmcov_vcc.cc			\
	specest_stream_to_vector_overlap.cc 	\
	specest_adaptiveweighting_vff.cc	\
	specest_moving_average_vff.cc 		\
	specest_reciprocal_ff.cc 		\
	specest_pad_vector.cc 			\
	specest_welch.cc 			\
	specest_burg.cc				\
	specest_fcov.cc				\
	specest_fmcov.cc

libgnuradio_specest_la_LIBADD = \
	$(GNURADIO_CORE_LA)

libgnuradio_specest_la_LDFLAGS = \
	$(NO_UNDEFINED)

# ----------------------------------------------------------------
# specest C++ QA library: libgnuradio-specest-qa.so (not installed)
# ----------------------------------------------------------------

noinst_LTLIBRARIES = libgnuradio-specest-qa.la

libgnuradio_specest_qa_la_SOURCES = 	\
	qa_specest.cc 			\
	qa_specest_arburg.cc

libgnuradio_specest_qa_la_LDFLAGS = $(NO_UNDEFINED)

libgnuradio_specest_qa_la_LIBADD = \
	libgnuradio-specest.la \
	$(CPPUNIT_LIBS)

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
noinst_HEADERS = 		\
	qa_specest.h 		\
	qa_specest_arburg.h

# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS = \
	test_all

test_all_SOURCES = test_all.cc
test_all_LDADD   = libgnuradio-specest-qa.la
