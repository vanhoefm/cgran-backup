#!/usr/bin/env python

#
# Copyright 2004,2005 Free Software Foundation, Inc.
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

from gnuradio import gr, blks
from gnuradio.eng_option import eng_option
from optparse import OptionParser
import random
import struct
import time
from bbn_80211b_pkt import *
from bbn_80211b import *

def rx_callback(ok, payload):
    size = struct.calcsize("@qHBB");
    packet_data = payload[size:];
    print packet_data

class my_graph(gr.flow_graph):

    def __init__(self, rx_callback, spb, alpha, SNR):
        # m is constellation size
        # if diff==True we are doing DxPSK
        gr.flow_graph.__init__(self)

        fg = self

        # transmitter
        self.packet_transmitter = bbn_80211b_mod_pkts(fg, spb=spb, alpha=alpha,
                                                      gain=1)

        # add some noise
        add = gr.add_cc()
        noise = gr.noise_source_c(gr.GR_GAUSSIAN, pow(10.0,-SNR/20.0))

        # channel filter
        rx_filt_taps = gr.firdes.low_pass(1,spb,0.8,0.1,gr.firdes.WIN_HANN)
        rx_filt = gr.fir_filter_ccf(1,rx_filt_taps)

        # receiver
        self.bit_receiver = bbn_80211b_demod_pkts(self, spb=spb, alpha=alpha,
                                                  callback=rx_callback)

        fg.connect(self.packet_transmitter, (add,0))
        fg.connect(noise, (add,1))

        #xfile=gr.file_sink(gr.sizeof_gr_complex,"txdata");
        #fg.connect(add, xfile)

        fg.connect(add, rx_filt)
        fg.connect(rx_filt, self.bit_receiver)


class stats(object):
    def __init__(self):
        self.npkts = 0
        self.nright = 0
        
def main():
    st = stats()
    
    def send_pkt(payload='', eof=False):
        fg.packet_transmitter.send_pkt(payload, eof)

    parser = OptionParser (option_class=eng_option)
    parser.add_option("","--spb", type=int, default=8,
                      help="set samples per baud to SPB [default=%default]")
    parser.add_option("", "--alpha", type="eng_float", default=0.4,
                      help="set excess bandwidth for RRC filter [default=%default]")
    parser.add_option("", "--snr", type="eng_float", default=12,
                      help="set SNR in dB for simulation [default=%default]")

    (options, args) = parser.parse_args ()

    if len(args) != 0:
        parser.print_help()
        sys.exit(1)

    fg = my_graph(rx_callback, options.spb, options.alpha, options.snr)

    fg.start()

    n = 0
    pktno = 0

    send_pkt('Hello World!')
    send_pkt('Hello Again!')
    send_pkt('The quick brown fox jumps over the lazy dog.')
    send_pkt(str(time.localtime()))

    send_pkt(eof=True) # tell modulator we're not sending any more pkts

    fg.wait()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
