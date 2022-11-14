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

from gnuradio import gr, gru, blks2
from gnuradio import usrp2
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from bbn_80211b_pkt import *
from optparse import OptionParser

import random
import time
import struct
import sys

n2s = eng_notation.num_to_str


class bbn_80211b_transmit_path(gr.top_block):

    def __init__(self, interp_rate, spb, use_barker, interface="", mac_addr=None):
        gr.top_block.__init__(self)
        
        self.normal_gain = 28000

        self.u = usrp2.sink_32fc(interface, mac_addr)
        dac_rate = self.u.dac_rate();
        self.set_gain(self.u.gain_max())  # set max Tx gain
        self._spb = spb
        self._interp=int(interp_rate)
        self.u.set_interp(self._interp)

        # transmitter
        self.packet_transmitter = bbn_80211b_mod_pkts(spb=spb,
                                                  alpha=0.5,
                                                  gain=self.normal_gain,
                                                  use_barker=use_barker)
        self.connect(self.packet_transmitter, self.u)      

    def set_freq(self, target_freq):
        """
        Set the center frequency we're interested in.

        @param target_freq: frequency in Hz
        @rypte: bool

        Tuning is a two step process.  First we ask the front-end to
        tune as close to the desired frequency as it can.  Then we use
        the result of that operation and our target_frequency to
        determine the value for the digital up converter.  Finally, we feed
        any residual_freq to the s/w freq translater.
        """
        tr = self.u.set_center_freq(target_freq)
        if tr == None:
        	sys.stderr.write('Failed to set center frequency\n')
        	raise SystemExit, 1
                        
        return tr

    def set_gain(self, gain):
        self.gain = gain
        self.u.set_gain(gain)
          
    def send_pkt(self, payload='', send_rate=0, eof=False):
        return self.packet_transmitter.send_pkt(payload, send_rate, eof)
          
    def spb(self):
        return self._spb

    def interp(self):
        return self._interp


# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():

    def send_pkt(payload='', send_rate=0, eof=False):
        tb.send_pkt(payload=payload, send_rate=send_rate, eof=eof)

    parser = OptionParser (option_class=eng_option)
    parser.add_option("-e", "--interface", type="string", default="eth0",
            help="use specified Ethernet interface [default=%default]")
    parser.add_option("-m", "--mac-addr", type="string", default="",
            help="use USRP2 at specified MAC address [default=None]")
    parser.add_option("-f", "--freq", type="eng_float", default=2.4e9,
                       help= \
                      "set Tx and Rx frequency to FREQ [default=%default]",
                      metavar="FREQ")
    parser.add_option("-S", "--spb", type="int", default=25,
                      help="set samples/baud [default=%default]")
    parser.add_option("-i", "--interp", type="int", default=4,
                      help=
                      "set fpga interpolation rate to INTERP [default=%default]")
    parser.add_option("-r", "--reps", type="int", default=10,
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
    tb = bbn_80211b_transmit_path(options.interp, options.spb, options.barker, options.interface, options.mac_addr)

    tr = tb.set_freq(options.freq)
            
    usrp = tb.u
    
    print ""
    print "Network interface: ", options.interface
    print "USRP2 address: ", usrp.mac_addr()
    print "Using TX d'board id 0x%04X" % (usrp.daughterboard_id(),)
    print "Tx baseband frequency: ", n2s(tr.baseband_freq)
    print "Tx DUC frequency: ", n2s(tr.dxc_freq)
    print "Tx residual frequency: ", n2s(tr.residual_freq)
    print "Tx interpolation rate: ", usrp.interp()
    print "Samples per data bit: ", tb.spb()
    if options.barker == True:
        print "Using Barker spreading"
 
    tb.start()                       # start flow graph

    # generate and send packets
    
    ############!!!To send packets, read the send_pkts function in the bbn_80211b_pkt.py file to understand the setup. . .
    n = 0
    bit_rate = 1
    fp = open('getty.txt')
    lines = fp.readlines()
    payload = "Hello world"
    i = 0;
    while i < len(lines):
        payload = payload + lines[i]
        i = i + 1
    
    print ""
    print ""    
    print "DATA:"
    print ""
    print payload
    print ""

    while n < options.reps:
        print "Sending pkt ", n
        send_pkt(payload + " " + str(n), bit_rate, False);
        n = n + 1

    time.sleep(1)
    send_pkt(eof=True)
    tb.wait()                       # wait for it to finish

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit
