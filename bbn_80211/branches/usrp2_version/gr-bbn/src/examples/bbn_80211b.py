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

class bbn_80211b_mod(gr.hier_block2):

  def __init__(self, bpsk=True):
    """
      Hierarchical block for RRC-filtered PSK modulation
      modulation.

      The input is a byte stream (unsigned char) and the
      output is the complex modulated signal at baseband.

      @param spb: samples per baud >= 2
      @type spb: integer
    @param alpha: Root-raised cosine filter excess bandwidth
    @type alpha: float
    """
    
    constellation = ( (),
                      ( 1-0j,-1+0j ),
                      ( 0.707+0.707j,-0.707-0.707j ),
                      ( 0.707+0.707j,-0.707+0.707j,-0.707-0.707j, 0.707-0.707j ),
                      ( -1+0j,-1j, 1j, 1+0j),
                      ( 1+0j,0-1j,-1+0j,0+1j ),
                      ( 1+0j,0+1j,-1+0j,0-1j ),
                      ( 0+0j,1+0j ),
                      ( 0+1j, 0-1j),
                      ( 0-1j, 0+1j)
                      )
    


    self.bpsk = bpsk

    if bpsk:
      self.bits_per_chunk = 1 #
      constellation_size = 2
      self.chunks2symbols = gr.chunks_to_symbols_bc(constellation[2]) #
      self.grey_code = [0, 1]
    else:
      self.bits_per_chunk = 2
      constellation_size = 4
      self.chunks2symbols = gr.chunks_to_symbols_bc(constellation[3]) #
      self.grey_code = [0, 1, 3, 2]
      
    self.scrambler = bbn.scrambler_bb(True)
    self.diff_encode = gr.diff_encoder_bb(constellation_size)  #
    self.bytes2chunks = gr.packed_to_unpacked_bb(self.bits_per_chunk, gr.GR_MSB_FIRST)
    self.grey_symbol = gr.map_bb(self.grey_code)

    
  
    
    gr.hier_block2.__init__(self, "bbn_80211b", gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_gr_complex))
    ### sets up the message sources and sinks for extracting and modifying the data with python
    self.bytes2chunks_src = gr.message_source(gr.sizeof_char, 5)   
    self.diff_encoded_src = gr.message_source(gr.sizeof_char, 5) 
    
    self.diff_encoded_queue = gr.msg_queue(5)
    self.diff_encoded_sink = gr.message_sink(gr.sizeof_char, self.diff_encoded_queue, False)    
    self.bytes2chunks_queue = gr.msg_queue(5)
    self.bytes2chunks_sink = gr.message_sink(gr.sizeof_char, self.bytes2chunks_queue, False)
# Connect

    self.connect(self, self.bytes2chunks, self.grey_symbol, self.bytes2chunks_sink)
    self.connect(self.bytes2chunks_src, self.diff_encode, self.diff_encoded_sink)  
    self.connect(self.diff_encoded_src, self.chunks2symbols, self)
    


# Initialize base class
    
    bbn.crc16_init()


class bbn_80211b_demod(gr.hier_block2):
  def __init__(self, pkt_queue, spb, alpha, use_barker=0, check_crc=False):
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
    self.plcp = bbn.plcp80211_bb(pkt_queue, check_crc);

    gr.hier_block2.__init__(self, "bbn_80211b_demod", gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(0, 0, 0))
    self.connect(self, self.rx_filter, self.slicer)
    self.connect(self.slicer, self.demod)
    self.connect((self.demod, 0), (self.plcp, 0));
    self.connect((self.demod, 1), (self.plcp, 1));

    bbn.crc16_init()
