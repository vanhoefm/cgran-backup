#
# Copyright 2004,2005,2006,2008,2009,2010 Free Software Foundation, Inc.
# 
# This file is part of GNU Radio
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
TESTS =

# ----------------------------------------------------------------
# itpp C++ library: libgnuradio-itpp.so
# ----------------------------------------------------------------

# C/C++ headers get installed in ${prefix}/include/$(modname)
modinclude_HEADERS = 		\
	itpp_bessel.h		\
	itpp_channel_tdl_vcc.h	\
	itpp_egolay.h		\
	itpp_hamming.h		\
	itpp_reedsolomon.h



lib_LTLIBRARIES = libgnuradio-itpp.la

libgnuradio_itpp_la_SOURCES =		\
	itpp_bessel.cc 			\
	itpp_channel_tdl_vcc.cc 	\
	itpp_egolay.cc 			\
	itpp_hamming.cc 		\
	itpp_reedsolomon.cc

libgnuradio_itpp_la_LIBADD = \
	$(GNURADIO_CORE_LA)

libgnuradio_itpp_la_LDFLAGS = 	\
	$(NO_UNDEFINED) 	\
	`itpp-config --libs`

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
noinst_HEADERS =

# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS =

test_all_SOURCES =
test_all_LDADD   =
