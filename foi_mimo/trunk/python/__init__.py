#
# Copyright 2011 FOI
# 
# Copyright 2008,2009 Free Software Foundation, Inc.
# 
# This application is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This application is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

# The presence of this file turns this directory into a Python package

# ----------------------------------------------------------------
# Temporary workaround for ticket:181 (swig+python problem)
import sys
_RTLD_GLOBAL = 0
try:
    from dl import RTLD_GLOBAL as _RTLD_GLOBAL
except ImportError:
    try:
	from DLFCN import RTLD_GLOBAL as _RTLD_GLOBAL
    except ImportError:
	pass
    
if _RTLD_GLOBAL != 0:
    _dlopenflags = sys.getdlopenflags()
    sys.setdlopenflags(_dlopenflags|_RTLD_GLOBAL)
# ----------------------------------------------------------------


# import swig generated symbols into the howto namespace
from foimimo_swig import *

# import any pure python here
from packet_mimo import options, packet_encoder, packet_decoder, \
        packet_mimo_mod_b, packet_mimo_mod_s, packet_mimo_mod_i, packet_mimo_mod_f, packet_mimo_mod_c, \
	packet_mimo_demod_b, packet_mimo_demod_s, packet_mimo_demod_i, packet_mimo_demod_f, packet_mimo_demod_c
from ofdm_mimo import ofdm_mimo_mod, ofdm_mimo_demod
from ofdm_mimo_with_coding import ofdm_mimo_mod_with_coding, ofdm_mimo_demod_with_coding
from ofdm_mimo_receiver import ofdm_mimo_receiver
from ofdm import ofdm_mod, ofdm_demod
from ofdm_with_coding import ofdm_mod_with_coding, ofdm_demod_with_coding
from ofdm_receiver import ofdm_receiver

# ----------------------------------------------------------------
# Tail of workaround
if _RTLD_GLOBAL != 0:
    sys.setdlopenflags(_dlopenflags)      # Restore original flags
# ----------------------------------------------------------------
