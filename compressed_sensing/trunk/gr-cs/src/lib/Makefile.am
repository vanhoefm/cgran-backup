#
# Copyright 2001-2008 Free Software Foundation, Inc.
# Copyright 2009 Institut fuer Nachrichtentechnik / Uni Karlsruhe
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

include $(top_srcdir)/Makefile.common

# Install this stuff so that it ends up as the gnuradio.cs module
# This usually ends up at:
#   ${prefix}/lib/python${python_version}/site-packages/gnuradio

ourpythondir = $(grpythondir)
ourlibdir    = $(grpyexecdir)

AM_CPPFLAGS = $(STD_DEFINES_AND_INCLUDES) $(PYTHON_CPPFLAGS) $(WITH_INCLUDES)

SWIG_PYTHON_ARGS = $(STD_SWIG_PYTHON_ARGS)

ALL_IFILES = 			\
	$(LOCAL_IFILES)		\
	$(NON_LOCAL_IFILES)

NON_LOCAL_IFILES =		\
	$(GNURADIO_I)

LOCAL_IFILES = 			\
	$(srcdir)/cs.i

# These files are built by SWIG.  The first is the C++ glue.
# The second is the python wrapper that loads the _cs shared library
# and knows how to call our extensions.
BUILT_SOURCES = 		\
	cs.cc			\
	cs.py

# This gets cs.py installed in the right place
ourpython_PYTHON =		\
	cs.py

ourlib_LTLIBRARIES = _cs.la

# These are the source files that go into the shared library (including SWIG file!)
_cs_la_SOURCES = 		\
	cs.cc			\
	cs_circmat_vccb.cc 	\
	cs_generic_vccf.cc 	\
	cs_nusaic_cc.cc 	\
	cs_toeplitz_vccb.cc

# magic flags
_cs_la_LDFLAGS = $(NO_UNDEFINED) -module -avoid-version

# link the library against some comon swig runtime code and the 
# c++ standard library
_cs_la_LIBADD = 		\
	$(PYTHON_LDFLAGS)	\
	-lstdc++

cs.cc cs.py: $(ALL_IFILES)
	$(SWIG) $(SWIG_PYTHON_ARGS) -module cs \
		-o cs.cc $(LOCAL_IFILES)

# These headers get installed in ${prefix}/include/gnuradio
grinclude_HEADERS =		\
	cs_circmat_vccb.h	\
	cs_generic_vccf.h	\
	cs_nusaic_cc.h 		\
	cs_toeplitz_vccb.h


# These swig headers get installed in ${prefix}/include/gnuradio/swig
swiginclude_HEADERS = 		\
	$(LOCAL_IFILES)


MOSTLYCLEANFILES = $(BUILT_SOURCES) *.pyc

# Don't distribute output of swig
dist-hook:
	@for file in $(BUILT_SOURCES); do echo $(RM) $(distdir)/$$file; done
	@for file in $(BUILT_SOURCES); do $(RM) $(distdir)/$$file; done
