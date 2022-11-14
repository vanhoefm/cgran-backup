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
if ARMADILLO
modinclude_HEADERS = \
	specesti_arburg.h 		\
	specesti_arfcov.h		\
	specesti_arfmcov.h		\
	specesti_correst.h		\
	specesti_esprit.h		\
	specesti_esprit_fortran.h	\
	specesti_music.h		\
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
	specest_stream_to_vector_overlap.h \
	specest_esprit_vcf.h		\
	specest_esprit_spectrum_vcf.h	\
	specest_esprit.h		\
	specest_music.h			\
	specest_music_vcf.h		\
	specest_music_spectrum_vcf.h \
	specesti_esprit_armadillo.h	\
	specesti_music_armadillo.h

else
modinclude_HEADERS = \
	specesti_arburg.h 		\
	specesti_arfcov.h		\
	specesti_arfmcov.h		\
	specesti_esprit.h		\
	specesti_esprit_fortran.h	\
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
	specest_stream_to_vector_overlap.h \
	specest_esprit_vcf.h		\
	specest_esprit_spectrum_vcf.h	\
	specest_esprit.h
endif !ARMADILLO

lib_LTLIBRARIES = libgnuradio-specest.la

if ARMADILLO
libgnuradio_specest_la_SOURCES = \
	specesti_arburg.cc 			\
	specesti_arfcov.cc			\
	specesti_arfmcov.cc			\
	specesti_correst.cc			\
	specesti_esprit_fortran.cc		\
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
	specest_fmcov.cc			\
	specest_esprit.cc			\
	specest_esprit_vcf.cc			\
	specest_esprit_spectrum_vcf.cc		\
	specest_music.cc			\
	specest_music_vcf.cc			\
	specest_music_spectrum_vcf.cc \
	specesti_esprit_armadillo.cc		\
	specesti_music_armadillo.cc
else
libgnuradio_specest_la_SOURCES = \
	specesti_arburg.cc 			\
	specesti_arfcov.cc			\
	specesti_arfmcov.cc			\
	specesti_esprit_fortran.cc		\
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
	specest_fmcov.cc			\
	specest_esprit.cc			\
	specest_esprit_vcf.cc \
	specest_esprit_spectrum_vcf.cc
endif !ARMADILLO


libgnuradio_specest_la_LIBADD = \
	$(GNURADIO_CORE_LA) \
	libgnuradio-specest-fortran.la \
	$(FLIBS)

libgnuradio_specest_la_LDFLAGS = \
	$(NO_UNDEFINED)

# ----------------------------------------------------------------
# specest Fortran library: libgnuradio-specest.so
# ----------------------------------------------------------------

libgnuradio_specest_fortran_la_SOURCES = \
	specesti_zcorrest.f90 \
	specesti_zesprit_common.f90 \
	specesti_zesprit.f90 \
	specesti_zesprit_spectrum.f90

libgnuradio_specest_fortran_la_LIBADD =

libgnuradio_specest_fortran_la_LDFLAGS = -llapack -lblas

# ----------------------------------------------------------------
# specest C++ QA library: libgnuradio-specest-qa.so (not installed)
# ----------------------------------------------------------------

noinst_LTLIBRARIES = libgnuradio-specest-qa.la \
	libgnuradio-specest-fortran.la

if ARMADILLO
libgnuradio_specest_qa_la_SOURCES = 	\
	qa_specest.cc 			\
	qa_specest_arburg.cc		\
	qa_specesti_esprit_fortran.cc	\
	qa_specesti_music_armadillo.cc \
	qa_specesti_esprit_armadillo.cc
else
libgnuradio_specest_qa_la_SOURCES = 	\
	qa_specest.cc 			\
	qa_specest_arburg.cc		\
	qa_specesti_esprit_fortran.cc
endif !ARMADILLO


libgnuradio_specest_qa_la_LDFLAGS = $(NO_UNDEFINED)

libgnuradio_specest_qa_la_LIBADD = \
	libgnuradio-specest.la \
	libgnuradio-specest-fortran.la \
	$(FLIBS) \
	$(CPPUNIT_LIBS)

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
if ARMADILLO
noinst_HEADERS = 		\
	qa_specest.h 		\
	qa_specest_arburg.h	\
	qa_specesti_esprit_fortran.h \
	qa_specesti_music_armadillo.h \
	qa_specesti_esprit_armadillo.h
else
noinst_HEADERS = 		\
	qa_specest.h 		\
	qa_specest_arburg.h	\
	qa_specesti_esprit_fortran.h
endif !ARMADILLO

# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS = \
	test_all

test_all_SOURCES = test_all.cc
test_all_LDADD   = libgnuradio-specest-qa.la
