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

from gnuradio import gr, gru
from gnuradio import usrp
from optparse import OptionParser
from bbn_80211b_pkt import *
from bbn_80211b_rx import *
from bbn_80211b_transmit_path import bbn_80211b_transmit_path

import sys
import struct
import os

class null_mac(object):
    def __init__(self, options):
        self.tap = bbn.bbn_make_tap("gr%d", int(options.freq / 1e6))
        self.tap_fd = self.tap.fd();

    def rx_callback(self, ok, payload):
        size = struct.calcsize("@qHBB");
        packet_data = payload[size:];
        self.tap.tap_write(payload)

class app_flow_graph(gr.flow_graph):
    def send_pkt(self, payload):
        self.txpath.send_pkt(payload)


    def __init__(self, callback, options):
        gr.flow_graph.__init__(self)

        if options.verbose:
            print "Samples per data bit = ", options.spb

        self.txpath = bbn_80211b_transmit_path(self,
                                               options.tx_subdev_spec,
                                               options.interp, options.spb,
                                               options.barker)

        ok = self.txpath.set_freq(options.freq)

        if not options.tx_only:
            self.u = usrp_rx(self, options.decim,
                             options.rx_subdev_spec, options.width_16,
                             options.verbose, options.gain,
                             options.freq)

            self.bit_rx = bbn_80211b_demod_pkts(self, spb=options.spb,
                                                alpha=0.5,
                                                callback=callback,
                                                use_barker=options.barker,
                                                check_crc=
                                                not options.no_crc_check)

            self.connect(self.u, self.bit_rx)
        self.freq=options.freq

def main ():
    parser = OptionParser(option_class=eng_option)
    parser.add_option("-R", "--rx-subdev-spec", type="subdev",
                      default=None,
                      help="select USRP Rx side A or B (default=first one with a daughterboard)")
    parser.add_option("-T", "--tx-subdev-spec", type="subdev",
                      default=None,
                      help="select USRP Tx side A or B")
    parser.add_option("-d", "--decim", type="int", default=8,
                      help="set fgpa decimation rate to DECIM [default=%default]")
    parser.add_option("-f", "--freq", type="eng_float", default=2437e6,
                      help="set frequency to FREQ", metavar="FREQ")
    parser.add_option("-g", "--gain", type="eng_float", default=None,
                      help="set gain in dB (default is midpoint)")
    parser.add_option("", "--width-16", action="store_true",
                      default=False,
                      help="Enable 16-bit samples across USB")
    parser.add_option("-S", "--spb", type="int", default=8, \
                      help="set samples/baud [default=%default]")
    parser.add_option("-b", "--barker", action="store_true",
                      default=True,
                      help="Use Barker Spreading [default=%default]")
    parser.add_option("-p", "--no-crc-check", action="store_true",
                      default=False,
                      help="Check payload crc [default=%default]")
    parser.add_option("-v", "--verbose", action="store_true",
                      default=False,
                      help="Verbose Output")
    parser.add_option("-i", "--interp", type="intx", default=32,
                      help=
                      "fpga interpolation rate [default=%default]")
    parser.add_option("-z", "--tx-only", action="store_true",
                      default=False,
                      help="Disable Receiver [default=%default]")
    parser.add_option("-t", "--tx-enable", action="store_true",
                      default=False,
                      help="Enable Transmitter [default=%default]")

    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        sys.exit(1)

    mac = null_mac(options)
    app = app_flow_graph(mac.rx_callback, options)
    app.start()

    while 1:
	payload = os.read(mac.tap.tap_read_fd(), 2400)
        payload = mac.tap.tap_process_tx(payload);
        if (len(payload) > 0) and (options.tx_enable):
          app.send_pkt(payload)

if __name__ == '__main__':
    main ()
