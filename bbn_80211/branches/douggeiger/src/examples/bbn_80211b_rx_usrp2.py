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
# Help making changes to work with hier_block2 from:
# http://1010.co.uk/bbn-examples-hier2.tar.gz

from gnuradio import gr, gru
from gnuradio import usrp2
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
import gnuradio.gr.gr_threading as _threading
from optparse import OptionParser
#from bbn_80211b_pkt import *
from bbn_80211b import *
import sys
import struct
import os

n2s = eng_notation.num_to_str

class _queue_watcher_thread(_threading.Thread):
    def __init__(self, rcvd_pktq, callback):
        _threading.Thread.__init__(self)
        self.setDaemon(1)
        self.rcvd_pktq = rcvd_pktq
        self.callback = callback
        self.keep_running = True
        self.start()

    def stop(self):
        self.keep_running = False
        
    def run(self):
        while self.keep_running:
            payload = self.rcvd_pktq.delete_head().to_string()
            if self.callback:
                self.callback(True, payload)


def rx_callback(ok, payload):
    size = struct.calcsize("@qHBB");
    packet_data = payload[size:];
    hdr = struct.unpack("@qHbB", payload[0:size]);
    if len(packet_data) > 16:
        data_hdr = struct.unpack("@BBBBBB", packet_data[10:16])
        mac_str = "%02x:%02x:%02x:%02x:%02X:%02X" % \
                  (data_hdr[0], data_hdr[1], data_hdr[2],
                   data_hdr[3], data_hdr[4], data_hdr[5],)
    else:
        mac_str = "UNKNOWN"
#    fd = open("pkt_data.txt", "a")
#    fd.write(payload);
#    fd.close()

    print "PKT: len=%d, rssi=%d, src=%s, time=%ld, rate=%d Mbps" \
          % (hdr[1], hdr[2], mac_str, hdr[0], hdr[3]/ 10.0)

class usrp2_rx(gr.hier_block2):
    def __init__(self, decim=16, verbose=False, gain=None, freq=None, interface="", mac_addr=None):

        gr.hier_block2.__init__(self, "usrp_rx",gr.io_signature(0,0,0),gr.io_signature(1,1,gr.sizeof_gr_complex))
        # build the graph
        self.u = usrp2.source_32fc(interface, mac_addr)
        # Set receiver decimation rate
        self.u.set_decim(decim)

        # Set receive daughterboard gain
        if gain is None:
            g = self.u.gain_range()
            gain = float(g[0]+g[1])/2
	    print "Using mid-point gain of", gain, "(", g[0], "-", g[1], ")"
        self.u.set_gain(gain)

        # Set receive frequency
        tr = self.u.set_center_freq(freq)
        if tr == None:
            sys.stderr.write('Failed to set center frequency\n')
            raise SystemExit, 1
	
	input_rate = self.u.adc_rate()/self.u.decim()
        
        if verbose:
            print "Network interface:", interface
            print "USRP2 address:", self.u.mac_addr()
            print "Using RX d'board id 0x%04X" % (self.u.daughterboard_id(),)
            print "Rx gain:", gain
            print "Rx baseband frequency:", n2s(tr.baseband_freq)
            print "Rx DDC frequency:", n2s(tr.dxc_freq)
            print "Rx residual frequency:", n2s(tr.residual_freq)
            print "Rx decimation rate:", decim
            print "Rx sample rate:", n2s(input_rate)

        self.connect(self.u,self)

class app_flow_graph(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)
        
        parser = OptionParser(option_class=eng_option)
	parser.add_option("-e", "--interface", type="string", default="eth0",
			help="use specified Ethernet interface [default=%default]")
	parser.add_option("-m", "--mac-addr", type="string", default="",
			help="use USRP2 at specified MAC address [default=None]")  
        parser.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
                          help="select USRP Rx side A or B (default=first one with a daughterboard)")
        parser.add_option("-d", "--decim", type="int", default=16,
                          help="set fgpa decimation rate to DECIM [default=%default]")
        parser.add_option("-f", "--freq", type="eng_float", default=2.4e9,
                          help="set frequency to FREQ", metavar="FREQ")
        parser.add_option("-g", "--gain", type="eng_float", default=None,
                          help="set gain in dB (default is midpoint)")
        parser.add_option("-S", "--spb", type="int", default=8, \
                          help="set samples/baud [default=%default]")
        parser.add_option("-b", "--barker", action="store_true",
                          default=False,
                          help="Use Barker Spreading [default=%default]")
        parser.add_option("-p", "--no-crc-check", action="store_true",
                          default=False,
                          help="Check payload crc [default=%default]")
        parser.add_option("-v", "--verbose", action="store_true",
                          default=False,
                          help="Verbose Output")

        (options, args) = parser.parse_args()
        if len(args) != 0:
            parser.print_help()
            sys.exit(1)

        self.u = usrp2_rx(decim=options.decim, verbose=options.verbose, gain=options.gain, freq=options.freq, interface=options.interface, mac_addr=options.mac_addr)

        if options.verbose:
            print "Samples per data bit = ", options.spb
        
#        self.bit_receiver = bbn_80211b_demod_pkts(callback=rx_callback, spb=options.spb, alpha=0.5)

        self._rcvd_pktq = gr.msg_queue()          # holds packets from the PHY

        self.bit_receiver = bbn_80211b_demod(spb=options.spb, alpha = 0.5, use_barker=options.barker, check_crc = not options.no_crc_check, pkt_queue=self._rcvd_pktq)

	self.connect(self.u, self.bit_receiver)

        self._watcher = _queue_watcher_thread(self._rcvd_pktq, rx_callback)

def main ():
    app = app_flow_graph()
    app.start()
    os.read(0,10)

if __name__ == '__main__':
    main ()
