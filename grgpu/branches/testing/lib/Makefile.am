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
# grgpu C++ library: libgnuradio-grgpu.so
# ----------------------------------------------------------------

# C/C++ headers get installed in ${prefix}/include/$(modname)
modinclude_HEADERS = \
 grgpu_fir_fff_cuda.h grgpu_add_const_ff_cuda.h           grgpu_resampler_fff_cuda.h grgpu_resampler_ccf_cuda.h   grgpu_fft_vfc_cuda.h grgpu_d2h_cuda.h grgpu_d2h_c_cuda.h grgpu_h2d_cuda.h grgpu_h2d_c_cuda.h grgpu_fir_filter_fff_cuda.h grgpu_utils.h



lib_LTLIBRARIES = libgnuradio-grgpu.la

libgnuradio_grgpu_la_SOURCES = \
 grgpu_fir_fff_cuda.cc grgpu_add_const_ff_cuda.cc           grgpu_fft_vfc_cuda.cc grgpu_resampler_fff_cuda.cc grgpu_resampler_ccf_cuda.cc   grgpu_d2h_cuda.cc grgpu_d2h_c_cuda.cc grgpu_h2d_cuda.cc grgpu_h2d_c_cuda.cc grgpu_fir_filter_fff_cuda.cc \
 grgpu_fir_fff_cuda_kernel.cu.cc grgpu_add_const_ff_cuda_kernel.cu.cc           grgpu_resampler_fff_cuda_kernel.cu.cc grgpu_resampler_ccf_cuda_kernel.cu.cc   grgpu_fft_vfc_cuda_kernel.cu.cc grgpu_d2h_cuda_kernel.cu.cc grgpu_d2h_c_cuda_kernel.cu.cc grgpu_h2d_cuda_kernel.cu.cc grgpu_h2d_c_cuda_kernel.cu.cc   grgpu_fir_filter_fff_cuda_kernel.cu.cc grgpu_utils.cu.cc

libgnuradio_grgpu_la_LIBADD = \
	$(GNURADIO_CORE_LA)

#WLP - augmented with general and specific cuda libraries 64 is listed
# first as a preference, but both it and the 32b cudart paths are
# listed as a quick compatibility fix.  (You may need to provide a
# symlink to the specific implementation of cutil: 
# e.g. ln -s libcutil_i386.a libcutil.a)
libgnuradio_grgpu_la_LDFLAGS = \
	$(NO_UNDEFINED) \
	-L/opt/local/bin -L$(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/ \
	-L/usr/local/cuda/lib64/ -L/usr/local/cuda/lib -lcudart  -lcufft -lcutil  

# ----------------------------------------------------------------
# grgpu C++ QA library: libgnuradio-grgpu-qa.so (not installed)
# ----------------------------------------------------------------

noinst_LTLIBRARIES = libgnuradio-grgpu-qa.la

libgnuradio_grgpu_qa_la_SOURCES = \
	qa_grgpu.cc \
 qa_grgpu_fir_fff_cuda.cc qa_grgpu_add_const_ff_cuda.cc           qa_grgpu_resampler_fff_cuda.cc qa_grgpu_resampler_ccf_cuda.cc   qa_grgpu_fft_vfc_cuda.cc  qa_grgpu_d2h_cuda.cc qa_grgpu_d2h_c_cuda.cc qa_grgpu_h2d_cuda.cc qa_grgpu_h2d_c_cuda.cc qa_grgpu_fir_filter_fff_cuda.cc \
 grgpu_fir_fff_cuda_kernel.cu.cc grgpu_add_const_ff_cuda_kernel.cu.cc           grgpu_resampler_fff_cuda_kernel.cu.cc grgpu_resampler_ccf_cuda_kernel.cu.cc   grgpu_fft_vfc_cuda_kernel.cu.cc  grgpu_d2h_cuda_kernel.cu.cc grgpu_d2h_c_cuda_kernel.cu.cc grgpu_h2d_cuda_kernel.cu.cc grgpu_h2d_c_cuda_kernel.cu.cc grgpu_fir_filter_fff_cuda_kernel.cu.cc grgpu_utils.cu.cc


#WLP - see linking note above
libgnuradio_grgpu_qa_la_LDFLAGS = $(NO_UNDEFINED) -version-info 0:0:0 \
	$(NO_UNDEFINED) \
	-L/opt/local/bin -L$(HOME)/NVIDIA_GPU_Computing_SDK/C/lib/ \
	-L/usr/local/cuda/lib64/ -L/usr/local/cuda/lib -lcudart -lcufft -lcutil


libgnuradio_grgpu_qa_la_LIBADD = \
	libgnuradio-grgpu.la \
	$(CPPUNIT_LIBS)

# ----------------------------------------------------------------
# headers that don't get installed
# ----------------------------------------------------------------
noinst_HEADERS = \
	qa_grgpu.h \
 qa_grgpu_fir_fff_cuda.h qa_grgpu_add_const_ff_cuda.h           qa_grgpu_resampler_fff_cuda.h qa_grgpu_resampler_ccf_cuda.h   qa_grgpu_fft_vfc_cuda.h qa_grgpu_d2h_cuda.h qa_grgpu_d2h_c_cuda.h qa_grgpu_h2d_cuda.h qa_grgpu_h2d_c_cuda.h qa_grgpu_fir_filter_fff_cuda.h 



# ----------------------------------------------------------------
# test program
# ----------------------------------------------------------------
noinst_PROGRAMS = \
	test_all

test_all_SOURCES = test_all.cc
test_all_LDADD   = libgnuradio-grgpu-qa.la
