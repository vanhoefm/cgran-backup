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
TESTS = test_all

# ----------------------------------------------------------------
# dabp C++ library: libgnuradio-dabp.so
# ----------------------------------------------------------------

# C/C++ headers get installed in ${prefix}/include/$(modname)
modinclude_HEADERS = \
	dabp_subchannel_selector.h dabp_time_deinterleaver.h \
    dabp_depuncturer.h dabp_vitdec.h dabp_scrambler.h \
    dabp_super_frame_sync.h dabp_firecode.h dabp_super_frame_rsdec.h \
    dabp_crc16.h dabp_super_frame_sink.h dabp_galois.h dabp_rscodec.h \
    dabp_moving_sum_ff.h dabp_ofdm_demod.h dabp_freq_interleaver.h \
    dabp_fic_msc_demux.h dabp_depuncturer_fic.h dabp_fib_sink.h \
    dabp_parameters.h

lib_LTLIBRARIES = libgnuradio-dabp.la

libgnuradio_dabp_la_SOURCES = \
	dabp_subchannel_selector.cc dabp_time_deinterleaver.cc \
    dabp_depuncturer.cc dabp_vitdec.cc dabp_scrambler.cc \
    dabp_super_frame_sync.cc dabp_firecode.cc dabp_super_frame_rsdec.cc \
    dabp_crc16.cc dabp_super_frame_sink.cc dabp_galois.cc dabp_rscodec.cc \
    dabp_moving_sum_ff.cc dabp_ofdm_demod.cc dabp_freq_interleaver.cc \
    dabp_fic_msc_demux.cc dabp_depuncturer_fic.cc dabp_fib_sink.cc \
    dabp_parameters.cc

libgnuradio_dabp_la_LIBADD = \
	$(GNURADIO_CORE_LA)

libgnuradio_dabp_la_LDFLAGS = $(NO_UNDEFINED) $(LTVERSIONFLAGS) -lgnuradio-trellis

# ----------------------------------------------------------------
# dabp C++ QA library: libgnuradio-dabp-qa.so (not installed)
# ----------------------------------------------------------------

noinst_LTLIBRARIES = libgnuradio-dabp-qa.la

libgnuradio_dabp_qa_la_SOURCES = \
    qa_dabp.cc

libgnuradio_dabp_qa_la_LDFLAGS = $(NO_UNDEFINED) -version-info 0:0:0

libgnuradio_dabp_qa_la_LIBADD = \
	libgnuradio-dabp.la \
	$(CPPUNIT_LIBS)

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
noinst_HEADERS = \
    qa_dabp.h

# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS = \
	test_all

test_all_SOURCES = test_all.cc
test_all_LDADD   = libgnuradio-dabp-qa.la
