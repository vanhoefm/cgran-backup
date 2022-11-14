#
# Copyright 2004,2005,2006,2008 Free Software Foundation, Inc.
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

INCLUDES = 	\
	$(STD_DEFINES_AND_INCLUDES) $(DEFINES) $(OMNITHREAD_INCLUDES) $(PMT_INCLUDES) $(MBLOCK_INCLUDES) \
	$(USRP_INCLUDES) $(USRP_INBAND_INCLUDES) $(BOOST_CFLAGS) $(CPPUNIT_INCLUDES) 

EXTRA_DIST =				\
	cmac.mbh			\
	tmac.mbh			\
	gmsk.mbh

lib_LTLIBRARIES = 			\
	libcmusdrg_macs.la	

GUILE_LOAD_PATH=GUILE_LOAD_PATH="$(top_srcdir)/src/scheme/pmt:$(top_srcdir)/src/scheme/mblock"
GUILE=/usr/bin/guile -e main -s $(top_srcdir)/src/scheme/mblock/gnuradio/compile-mbh.scm

# ------------------------------------------------------------------------
# Build the inband library

BUILT_SOURCES =				\
	cmac_mbh.cc		\
	tmac_mbh.cc		\
	gmsk_mbh.cc 

cmac_mbh.cc : cmac.mbh
	$(GUILE_LOAD_PATH) $(GUILE) $(srcdir)/cmac.mbh $(srcdir)/cmac_mbh.cc

tmac_mbh.cc : tmac.mbh
	$(GUILE_LOAD_PATH) $(GUILE) $(srcdir)/tmac.mbh $(srcdir)/tmac_mbh.cc

gmsk_mbh.cc : gmsk.mbh
	$(GUILE_LOAD_PATH) $(GUILE) $(srcdir)/gmsk.mbh $(srcdir)/gmsk_mbh.cc

libcmusdrg_macs_la_SOURCES =		\
	$(BUILT_SOURCES)		\
	mac.cc \
	cmac.cc \
	cmac_framer.cc \
	tmac.cc	\
	tmac_framer.cc \
	gmsk.cc 

libcmusdrg_macs_la_LDFLAGS = $(NO_UNDEFINED) -version-info 0:0:0

libcmusdrg_macs_la_LIBADD =		\
	$(USRP_LA)	  		\
	-lstdc++					\
	-lusrp \
	-lusrp_inband		\
	-lmblock

include_HEADERS =			\
	mac.h \
	cmac.h \
	cmac_framer.h \
	tmac.h	\
	tmac_framer.h \
	gmsk.h 


noinst_HEADERS =      			\
	gmsk_symbols.h	\
	tmac_symbols.h	\
	cmac_symbols.h


MOSTLYCLEANFILES = \
	$(BUILT_SOURCES) *~ *.pyc
