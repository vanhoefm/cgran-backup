#
# Copyright 2003,2006 Free Software Foundation, Inc.
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
	$(USRP_INCLUDES) $(USRP_INBAND_INCLUDES) $(BOOST_CFLAGS) $(CPPUNIT_INCLUDES) \
	-I$(srcdir)/../lib

bin_PROGRAMS =				

noinst_PROGRAMS =			\
	tmac_basestation \
	tx_file	\
	rx_file


tx_file_SOURCES 	= tx_file.cc 
tx_file_LDADD   	= $(USRP_LA) $(USRP_INBAND_LA) ../lib/libcmusdrg_macs.la

rx_file_SOURCES 	= rx_file.cc 
rx_file_LDADD   	= $(USRP_LA) $(USRP_INBAND_LA) ../lib/libcmusdrg_macs.la

tmac_basestation_SOURCES 	= tmac_basestation.cc 
tmac_basestation_LDADD   	= $(USRP_LA) $(USRP_INBAND_LA) ../lib/libcmusdrg_macs.la
