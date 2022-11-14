#!/usr/bin/env python
#
# Copyright 2004,2005 Free Software Foundation, Inc.
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

from gnuradio import gr, gru
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser
from bbn_80211b_pkt import *
from bbn_80211b import *
import sys
import struct
import os

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
             
def pick_subdevice(u):
    """
    The user didn't specify a subdevice on the command line.
    If there's a daughterboard on A, select A.
    If there's a daughterboard on B, select B.
    Otherwise, select A.
    """
    if u.db[0][0].dbid() >= 0: # dbid is < 0 if there's no d'board or a problem
        return (0, 0)
    if u.db[1][0].dbid() >= 0:
        return (1, 0)
    return (0, 0)


class usrp_rx(gr.hier_block):
    def __init__(self, fg, decim=8, rx_subdev_spec=None, width_16=False,
                 verbose=False, gain=None, freq=None):
        # build the graph"
        self.u = usrp.source_c(decim_rate=decim, fpga_filename="usrp_std_d2.rbf")
        if rx_subdev_spec is None:
            rx_subdev_spec = pick_subdevice(self.u)

        self.u.set_mux(usrp.determine_rx_mux_value(self.u, rx_subdev_spec))

        if not width_16:
            width = 8
            shift = 8
            format = self.u.make_format(width, shift)
            r = self.u.set_format(format)
            if verbose:
                print "Bits Per Encoded Sample = 8"
        else:
            if verbose:
                print "Bits Per Encoded Sample = 16"
        
        # determine the daughterboard subdevice we're using
        self.subdev = usrp.selected_subdev(self.u, rx_subdev_spec)

        if verbose:
            print "adc frequency = ", self.u.adc_freq()
            print "decimation frequency = ", self.u.decim_rate()
            print "input_rate = ", self.u.adc_freq() / self.u.decim_rate()

        if gain is None:
            # if no gain was specified, use the mid-point in dB
            g = self.subdev.gain_range()
            gain = float(g[0]+g[1])/2

        if verbose:
            print "gain = ", gain

        if freq is None:
            # if no freq was specified, use the mid-point
            r = self.subdev.freq_range()
            freq = float(r[0]+r[1])/2

        self.subdev.set_gain(gain)
        r = self.u.tune(0, self.subdev, freq)

        if verbose:
            print "desired freq = ", freq
            #print "baseband frequency", r.baseband_freq
            #print "dxc frequency", r.dxc_freq

        gr.hier_block.__init__(self, fg, None, self.u)

class app_flow_graph(gr.flow_graph):
    def __init__(self):
        gr.flow_graph.__init__(self)
        
        parser = OptionParser(option_class=eng_option)
        parser.add_option("-R", "--rx-subdev-spec", type="subdev",
                          default=None,
                          help="select USRP Rx side A or B (default=first one with a daughterboard)")
        parser.add_option("-d", "--decim", type="int", default=8,
                          help="set fgpa decimation rate to DECIM [default=%default]")
        parser.add_option("-f", "--freq", type="eng_float", default=2.4e9,
                          help="set frequency to FREQ", metavar="FREQ")
        parser.add_option("-g", "--gain", type="eng_float", default=None,
                          help="set gain in dB (default is midpoint)")
        parser.add_option("", "--width-16", action="store_true",
                          default=False,
                          help="Enable 16-bit samples across USB")
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

        self.u = usrp_rx(self, options.decim, options.rx_subdev_spec,
                         options.width_16, options.verbose, options.gain,
                         options.freq)

        if options.verbose:
            print "Samples per data bit = ", options.spb
        
        self.bit_receiver = bbn_80211b_demod_pkts(self, spb=options.spb,
                                                  alpha=0.5,
                                                  callback=rx_callback,
                                                  use_barker=options.barker,
                                                  check_crc=
                                                  not options.no_crc_check)

        self.connect(self.u, self.bit_receiver)

def main ():
    app = app_flow_graph()
    app.start()
    os.read(0,10)

if __name__ == '__main__':
    main ()
