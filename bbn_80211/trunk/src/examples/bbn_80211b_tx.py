#!/usr/bin/env python
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

from gnuradio import gr, gru, blks
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser

import random
import time
import struct
import sys

# from current dir
from bbn_80211b_transmit_path import bbn_80211b_transmit_path

class my_graph(gr.flow_graph):

    def __init__(self, tx_subdev_spec, interp_rate, spb, use_barker):
        gr.flow_graph.__init__(self)
        self.txpath = bbn_80211b_transmit_path(self, tx_subdev_spec, \
                                               interp_rate, spb, use_barker)


# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():

    def send_pkt(payload='', eof=False):
        return fg.txpath.send_pkt(payload, eof)

    def rx_callback(ok, payload):
        print "ok = %r, payload = '%s'" % (ok, payload)

    parser = OptionParser (option_class=eng_option)
    parser.add_option("-T", "--tx-subdev-spec", type="subdev", default=None,
                      help="select USRP Tx side A or B")
    parser.add_option("-f", "--freq", type="eng_float", default=2.4e9,
                       help= \
                      "set Tx and Rx frequency to FREQ [default=%default]",
                      metavar="FREQ")
    parser.add_option("-S", "--spb", type="int", default=8,
                      help="set samples/baud [default=%default]")
    parser.add_option("-i", "--interp", type="int", default=32,
                      help=
                      "set fpga interpolation rate to INTERP [default=%default]")
    parser.add_option("-r", "--reps", type="int", default=20,
                      help=
                      "Number of packets to send [default=%default]")
    parser.add_option("-b", "--barker", action="store_true",
                      default=False,
                      help="Use Barker Spreading [default=%default]")

    (options, args) = parser.parse_args ()

    if len(args) != 0:
        parser.print_help()
        sys.exit(1)

    if options.freq < 1e6:
        options.freq *= 1e6

    # build the graph
    fg = my_graph(options.tx_subdev_spec, options.interp, options.spb, \
                  options.barker)

    #print "bitrate: %sb/sec" % (eng_notation.num_to_str(fg.txpath.bitrate()),)
    print "spb:     %3d" % (fg.txpath.spb(),)
    print "interp:  %3d" % (fg.txpath.interp(),)

    ok = fg.txpath.set_freq(options.freq)
    if not ok:
        print "Failed to set Tx frequency to %s" % (eng_notation.num_to_str(options.freq),)
        raise SystemExit

    fg.start()                       # start flow graph

    # generate and send packets
    n = 0

    fp = open('getty.txt')
    lines = fp.readlines()
    payload = ""
    i = 0;
    while i < len(lines):
        payload = payload + lines[i]
        i = i + 1

    while n < options.reps:
        send_pkt(payload, False);
        n = n + 1

    time.sleep(1)
    send_pkt(eof=True)
    fg.wait()                       # wait for it to finish

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
