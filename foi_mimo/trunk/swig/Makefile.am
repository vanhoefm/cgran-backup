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

AM_CPPFLAGS += -I$(top_srcdir)/lib

if PYTHON
###################################
# SWIG Python interface and library

TOP_SWIG_IFILES = \
	foimimo.i \
	fsm.i

# Install so that they end up available as:
#   import howto
# This ends up at:
#   ${prefix}/lib/python${python_version}/site-packages/$(modname)

foimimo_la_swig_libadd =		\
	$(top_builddir)/lib/libgnuradio-foimimo.la

include $(top_srcdir)/Makefile.swig

# add some of the variables generated inside the Makefile.swig.gen
BUILT_SOURCES = $(swig_built_sources)

# Do not distribute the output of SWIG
no_dist_files = $(swig_built_sources)

# additional SWIG files to be installed
foimimo_swiginclude_headers = \
	foimimo_ofdm_alamouti_frame_acquisition.i \
	foimimo_ofdm_alamouti_tx_cc.i \
	foimimo_ofdm_mapper_bcv.i \
	foimimo_ofdm_mimo_frame_sink.i	

endif
