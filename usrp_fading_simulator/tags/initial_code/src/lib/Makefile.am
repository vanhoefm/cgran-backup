#
# Copyright 2004,2005,2006 Free Software Foundation, Inc.
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
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

include $(top_srcdir)/Makefile.common

# Install this stuff so that it ends up as the gnuradio.tait module
# This usually ends up at:
#   ${prefix}/lib/python${python_version}/site-packages/gnuradio

ourpythondir = $(grpythondir)
ourlibdir    = $(grpyexecdir)

INCLUDES = $(STD_DEFINES_AND_INCLUDES) $(PYTHON_CPPFLAGS)

SWIGPYTHONARGS = $(SWIGPYTHONFLAGS) $(SWIGGRFLAGS)

ALL_IFILES = 				\
	$(LOCAL_IFILES)			\
	$(NON_LOCAL_IFILES)		

NON_LOCAL_IFILES =			\
	$(GNURADIO_CORE_INCLUDEDIR)/swig/gnuradio.i


LOCAL_IFILES = 				\
	$(top_srcdir)/src/lib/tait.i

#----------------------------- Existing DSP C-code Locations -----------------------------
# The directory where the c source code is located.
C_SOURCE =					\
	c_code

# Location of files which are common to a number of different blocks.
COMMON_SOURCE =					\
	common_source
# Location (directory) of the C4FM detection .c and .h files.
C4FM_DETECT_SOURCE =				\
	c4fm_detect
	
# Location (directory) of the biquad fillter .c and .h files.
BIQUAD4_SOUREC =				\
	biquad4

# Location (directory) of the example .c and .h files.
EXAMPLE_SOURCE =				\
	example
	
# Location (directory) of the DC corrector .c and .h files.
DC_CORRECTOR_SOURCE =				\
	DC_corrector
	
# Location (directory) of the fading .cc and .h files.
FADING_SOURCE =				\
	fading_channel
#-----------------------------------------------------------------------------------------


# These files are built by SWIG.  The first is the C++ glue.
# The second is the python wrapper that loads the _tait shared library
# and knows how to call our extensions.

BUILT_SOURCES = 		\
	tait.cc			\
	tait.py	

# This gets tait.py installed in the right place
ourpython_PYTHON =		\
	tait.py

ourlib_LTLIBRARIES = _tait.la

# These are the source files that go into the shared library
_tait_la_SOURCES = 							\
	tait.cc								\
	tait_c4fm_detect_s.cc						\
	tait_biquad4_ss.cc						\
	tait_example_ff.cc						\
	tait_DC_corrector_ss.cc						\
	tait_DC_corrector_ff.cc						\
	tait_socket_encode_fchar.cc					\
	tait_socket_encode_schar.cc					\
	tait_flat_rayleigh_channel_cc.cc				\
	$(FADING_SOURCE)/flat_rayleigh.cpp				\
	$(FADING_SOURCE)/Random.cpp					\
	$(C_SOURCE)/$(EXAMPLE_SOURCE)/example.c				\
	$(C_SOURCE)/$(COMMON_SOURCE)/etsi_fp_math.c			\
	$(C_SOURCE)/$(COMMON_SOURCE)/DSP_C_code_interface.c		\
	$(C_SOURCE)/$(C4FM_DETECT_SOURCE)/spapmd_c4fmDetect3.c		\
	$(C_SOURCE)/$(C4FM_DETECT_SOURCE)/spapmd_goertzel_c.c		\
	$(C_SOURCE)/$(BIQUAD4_SOUREC)/lib_biquad4.c			\
	$(C_SOURCE)/$(DC_CORRECTOR_SOURCE)/spas_dcCorrector.c			
	

# magic flags
_tait_la_LDFLAGS = $(NO_UNDEFINED) -module -avoid-version

# link the library against some comon swig runtime code and the 
# c++ standard library
_tait_la_LIBADD = 			\
	$(PYTHON_LDFLAGS)		\
	-lstdc++			

tait.cc tait.py: $(LOCAL_IFILES) $(ALL_IFILES)
	$(SWIG) $(SWIGPYTHONARGS) -module tait -o tait.cc $(LOCAL_IFILES)

# These headers get installed in ${prefix}/include/gnuradio
grinclude_HEADERS =							\
	tait_c4fm_detect_s.h						\
	tait_biquad4_ss.h						\
	tait_example_ff.h						\
	tait_DC_corrector_ss.h						\
	tait_DC_corrector_ff.h						\
	tait_socket_encode_fchar.h					\
	tait_socket_encode_schar.h					\
	tait_flat_rayleigh_channel_cc.h					\
	$(FADING_SOURCE)/flat_rayleigh.h				\
	$(FADING_SOURCE)/Random.h					\
	$(FADING_SOURCE)/Complex.h					\	
	$(C_SOURCE)/$(EXAMPLE_SOURCE)/example.h				\
	$(C_SOURCE)/$(COMMON_SOURCE)/etsi_fp_math.h			\
	$(C_SOURCE)/$(COMMON_SOURCE)/system_stddefsPC.h			\
	$(C_SOURCE)/$(COMMON_SOURCE)/DSP_C_code_interface.h		\
	$(C_SOURCE)/$(COMMON_SOURCE)/UserSim.h				\
	$(C_SOURCE)/$(COMMON_SOURCE)/simUtils.h				\
	$(C_SOURCE)/$(C4FM_DETECT_SOURCE)/spapmd_c4fmDetect_s.h		\
	$(C_SOURCE)/$(C4FM_DETECT_SOURCE)/spapmd_c4fmDetect_ss.h	\
	$(C_SOURCE)/$(BIQUAD4_SOUREC)/lib_biquad4_s.h			\
	$(C_SOURCE)/$(DC_CORRECTOR_SOURCE)/spas_dcCorrector.h		
	
				

# These swig headers get installed in ${prefix}/include/gnuradio/swig
swiginclude_HEADERS = 			\
	$(LOCAL_IFILES)


MOSTLYCLEANFILES = $(BUILT_SOURCES) *.pyc

# Don't distribute output of swig
dist-hook:
	@for file in $(BUILT_SOURCES); do echo $(RM) $(distdir)/$$file; done
	@for file in $(BUILT_SOURCES); do $(RM) $(distdir)/$$file; done
