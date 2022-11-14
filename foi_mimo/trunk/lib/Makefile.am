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
# howto C++ library: libgnuradio-howto.so
# ----------------------------------------------------------------

# C/C++ headers get installed in ${prefix}/include/$(modname)
modinclude_HEADERS = \
	foimimo_ofdm_alamouti_frame_acquisition.h \
	foimimo_ofdm_alamouti_tx_cc.h \
	foimimo_ofdm_mapper_bcv.h \
	foimimo_ofdm_mapper_source.h \
	foimimo_ofdm_mimo_frame_sink.h \
	foimimo_crc32_checker_sink.h \
	foimimo_crc32_inserter.h \
	foimimo_descrambler_bb.h \
	foimimo_scrambler_bb.h \
	foimimo_trellis_encoder_bb.h \
	foimimo_trellis_viterbi_b.h \
	foimimo_ofdm_demapper.h \
	foimimo_trellis_metrics_f.h \
	foimimo_chunk_2_byte.h \
	foimimo_byte_2_chunk.h \
	foimimo_chunk_2_byte_skip_head.h \
        foimimo_ofdm_frame_acquisition.h \
        gri_lfsr.h \
        fsm.h \
        base.h

lib_LTLIBRARIES = libgnuradio-foimimo.la

libgnuradio_foimimo_la_SOURCES = \
	foimimo_ofdm_alamouti_frame_acquisition.cc \
	foimimo_ofdm_alamouti_tx_cc.cc \
	foimimo_ofdm_mapper_bcv.cc \
	foimimo_ofdm_mapper_source.cc \
	foimimo_ofdm_mimo_frame_sink.cc \
	foimimo_crc32_checker_sink.cc \
        foimimo_crc32_inserter.cc \
        foimimo_descrambler_bb.cc \
        foimimo_scrambler_bb.cc \
        foimimo_trellis_encoder_bb.cc \
        foimimo_trellis_viterbi_b.cc  \
        foimimo_ofdm_demapper.cc \
        foimimo_trellis_metrics_f.cc \
        foimimo_chunk_2_byte.cc \
        foimimo_byte_2_chunk.cc \
        foimimo_chunk_2_byte_skip_head.cc \
        foimimo_ofdm_frame_acquisition.cc \
        fsm.cc \
        base.cc

libgnuradio_foimimo_la_LIBADD = \
	$(GNURADIO_CORE_LA)

libgnuradio_foimimo_la_LDFLAGS = $(NO_UNDEFINED) $(LTVERSIONFLAGS)

# ----------------------------------------------------------------
# howto C++ QA library: libgnuradio-howto-qa.so (not installed)
# ----------------------------------------------------------------

noinst_LTLIBRARIES = libgnuradio-foimimo-qa.la

libgnuradio_foimimo_qa_la_SOURCES = 

libgnuradio_foimimo_qa_la_LDFLAGS = $(NO_UNDEFINED) -version-info 0:0:0

libgnuradio_foimimo_qa_la_LIBADD = \
	libgnuradio-foimimo.la \
	$(CPPUNIT_LIBS)

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
noinst_HEADERS =  

# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS = \
	test_all

test_all_SOURCES = test_all.cc
test_all_LDADD   = libgnuradio-foimimo-qa.la
