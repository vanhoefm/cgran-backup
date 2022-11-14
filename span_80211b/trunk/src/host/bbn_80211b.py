#!/usr/bin/env python

# 802.11 style bpsk modulation and demodulation.  
#
#
# Copyright 2005 Free Software Foundation, Inc.
#
# Copyright (c) 2006 BBN Technologies Corp.  All rights reserved.
# Effort sponsored in part by the Defense Advanced Research Projects
# Agency (DARPA) and the Department of the Interior National Business
# Center under agreement number NBCHC050166.
# 
# For implementation of a full bandwidth 802.11b receiver, it's been 
# Modified by Mohammad H. Firooz, SPAN Lab., 
# University of Utah, UT-84112, in 2008. 
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

# See gnuradio-examples/python/xpsk for examples

from gnuradio import gr
from gnuradio import bbn
from math import pi
import Numeric

# /////////////////////////////////////////////////////////////////////////////
#            mPSK mod/demod with steams of bytes as data i/o
# /////////////////////////////////////////////////////////////////////////////

class bbn_80211b_mod(gr.hier_block):

    def __init__(self, fg, spb, alpha, gain, use_barker=0):
        """
	Hierarchical block for RRC-filtered PSK modulation
	modulation.

	The input is a byte stream (unsigned char) and the
	output is the complex modulated signal at baseband.

	@param fg: flow graph
	@type fg: flow graph
	@param spb: samples per baud >= 2
	@type spb: integer
	@param alpha: Root-raised cosine filter excess bandwidth
	@type alpha: float
	"""
        if not isinstance(spb, int) or spb < 2:
            raise TypeError, "sbp must be an integer >= 2"
        self.spb = spb
        self.bits_per_chunk = 1

	ntaps = 2 * spb - 1
        alpha = 0.5

        # turn bytes into symbols
        self.bytes2chunks = gr.packed_to_unpacked_bb(self.bits_per_chunk,
                                                     gr.GR_MSB_FIRST)

        constellation = ( (),
                          ( -1-0j,1+0j ),
                          ( 0.707+0.707j,-0.707-0.707j ),
                          ( 0.707+0j,-0.707-0.707j ),
                          ( -1+0j,-1j, 1j, 1+0j),
                          ( 1+0j,0+1j,-1+0j,0-1j ),
                          ( 0+0j,1+0j ),
                          )

        self.chunks2symbols = gr.chunks_to_symbols_bc(constellation[2])
        self.scrambler = bbn.scrambler_bb(True)
        self.diff_encode = gr.diff_encoder_bb(2);

        self.barker_taps = bbn.firdes_barker(spb)

	# Form Raised Cosine filter
	self.rrc_taps = gr.firdes.root_raised_cosine(
		4 * gain,     	# gain  FIXME may need to be spb
		spb,            # sampling freq
		1.0,		# symbol_rate
		alpha,
                ntaps)

        if use_barker:
            self.tx_filter = gr.interp_fir_filter_ccf(spb, self.barker_taps)
        else:
            self.tx_filter = gr.interp_fir_filter_ccf(spb, self.rrc_taps)

	# Connect
        fg.connect(self.scrambler, self.bytes2chunks)
        fg.connect(self.bytes2chunks, self.diff_encode)
        fg.connect(self.diff_encode, self.chunks2symbols)
	fg.connect(self.chunks2symbols,self.tx_filter)

	# Initialize base class
        gr.hier_block.__init__(self, fg, self.scrambler, self.tx_filter)
        bbn.crc16_init()


class bbn_80211b_demod(gr.hier_block):
    def __init__(self, fg, pkt_queue, spb, alpha,  use_barker=0,
                 check_crc=True):
        # RRC data filter
	ntaps = 2 * spb - 1

	self.rrc_taps = gr.firdes.root_raised_cosine(
		1,		# gain  FIXME may need to be spb
		spb,             # sampling freq
		1.0,		# symbol_rate
		alpha,
                ntaps)

        self.barker_taps = bbn.firdes_barker(spb)

        if use_barker == 1:
            self.rx_filter = gr.fir_filter_ccf(1, self.barker_taps)
        else:
            self.rx_filter = gr.fir_filter_ccf(1, self.rrc_taps)

        self.slicer = bbn.slicer_cc(spb, 16);
        self.demod = bbn.dpsk_demod_cb();
        self.descramble = bbn.scrambler_bb(False);
	print "CRC Check is ", check_crc;
        self.plcp = bbn.plcp80211_bb(pkt_queue, check_crc);

	self.amp = gr.multiply_const_cc(1);

        #fg.connect(self.rx_filter, self.slicer);
        #fg.connect(self.slicer, self.demod);
        fg.connect(self.amp, self.demod);
        fg.connect((self.demod, 0), (self.plcp, 0));
        fg.connect((self.demod, 1), (self.plcp, 1));

        #gr.hier_block.__init__(self, fg, self.rx_filter, self.plcp)
	gr.hier_block.__init__(self, fg, self.amp, self.plcp)
        bbn.crc16_init()
