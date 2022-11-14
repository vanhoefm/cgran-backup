#
# Copyright 2004,2005 Free Software Foundation, Inc.
# Copyright (c) 2006 BBN Technologies Corp.  All rights reserved.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
# 

include $(top_srcdir)/Makefile.common

# Install this stuff so that it ends up as the gnuradio.bbn module
# This usually ends up at:
#   ${prefix}/lib/python${python_version}/site-packages/gnuradio

ourpythondir = $(grpythondir)
ourlibdir    = $(grpyexecdir)

INCLUDES = $(STD_DEFINES_AND_INCLUDES) $(PYTHON_CPPFLAGS)

SWIGCPPPYTHONARGS = -fvirtual -python -modern $(PYTHON_CPPFLAGS) \
	-I$(swigincludedir) -I$(grincludedir)

ALL_IFILES = 				\
	$(LOCAL_IFILES)			\
	$(NON_LOCAL_IFILES)		

NON_LOCAL_IFILES =			\
	$(GNURADIO_CORE_INCLUDEDIR)/swig/gnuradio.i


LOCAL_IFILES = 				\
	bbn.i

# These files are built by SWIG.  The first is the C++ glue.
# The second is the python wrapper that loads the _bbn shared library
# and knows how to call our extensions.

BUILT_SOURCES = 			\
	bbn.cc				\
	bbn.py				

# This gets bbn.py installed in the right place
ourpython_PYTHON =			\
	bbn.py

ourlib_LTLIBRARIES = _bbn.la

# These are the source files that go into the shared library
_bbn_la_SOURCES = 			\
	bbn.cc				\
	bbn_tap.cc			\
	bbn_dpsk_demod_cb.cc  		\
	bbn_slicer_cc.cc  		\
	bbn_crc16.cc  			\
	bbn_plcp80211_bb.cc		\
	bbn_scrambler_bb.cc		\
	bbn_firdes_barker.cc

# magic flags
_bbn_la_LDFLAGS = $(NO_UNDEFINED) -module -avoid-version

# link the library against some comon swig runtime code and the 
# c++ standard library
_bbn_la_LIBADD = 			\
	$(PYTHON_LDFLAGS)		\
	-lstdc++			

bbn.cc bbn.py: bbn.i $(ALL_IFILES)
	$(SWIG) $(SWIGCPPPYTHONARGS) -module bbn -o bbn.cc bbn.i

# These headers get installed in ${prefix}/include/gnuradio
grinclude_HEADERS =			\
	bbn_dpsk_demod_cb.h  		\
	bbn_slicer_cc.h  		\
	bbn_crc16.h  			\
	bbn_plcp80211_bb.h		\
	bbn_scrambler_bb.h		\
	bbn_firdes_barker.h		\
	bbn_tap.h


# These swig headers get installed in ${prefix}/include/gnuradio/swig
swiginclude_HEADERS = 			\
	bbn_dpsk_demod_cb.i  		\
	bbn_slicer_cc.i  		\
	bbn_crc16.i  			\
	bbn_plcp80211_bb.i		\
	bbn_scrambler_bb.i		\
	bbn_firdes_barker.i		\
	bbn_tap.i			\
	$(LOCAL_IFILES)


MOSTLYCLEANFILES = $(BUILT_SOURCES) *.pyc
