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
from gnuradio import usrp2
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
  print "PKT: len=%d, rssi=%d, src=%s, time=%ld, rate=%d Mbps" \
        % (hdr[1], hdr[2], mac_str, hdr[0], hdr[3]/ 10.0)
  print "Payload: ", packet_data
             

class usrp_rx(gr.hier_block2):
  def __init__(self,decim=4, verbose=False, gain=None, freq=None):
    gr.hier_block2.__init__(self, "usrp_rx", gr.io_signature(0, 0, 0), gr.io_signature(1, 2, gr.sizeof_gr_complex))

    self.u = usrp2.source_32fc("eth0", "")
    self.u.set_decim(decim)
      

    if verbose:
      print "adc frequency = ", self.u.adc_rate()
      print "decimation frequency = ", self.u.decim()
      print "input_rate = ", self.u.adc_rate() / self.u.decim()

    if gain is None:
      # if no gain was specified, use the mid-point in dB
      g = self.u.gain_range()
      gain = float(g[0]+g[1])/2

    if verbose:
      print "gain = ", gain

    if freq is None:
      # if no freq was specified, use the mid-point
      r = self.u.freq_range()
      freq = float(r[0]+r[1])/2

    self.u.set_gain(gain)
    r = self.u.set_center_freq(freq)

    if verbose:
      print "desired freq = ", freq
      print "baseband frequency", r.baseband_freq
      print "dxc frequency", r.dxc_freq
    
    self.connect(self.u, self)

class app_flow_graph(gr.top_block):
  def __init__(self):
    gr.top_block.__init__(self)
        
    parser = OptionParser(option_class=eng_option)
    parser.add_option("-d", "--decim", type="int", default=4, help="set fgpa decimation rate to DECIM [default=%default]")
    parser.add_option("-f", "--freq", type="eng_float", default=2.4e9, help="set frequency to FREQ", metavar="FREQ")
    parser.add_option("-g", "--gain", type="eng_float", default=None, help="set gain in dB (default is midpoint)")
    parser.add_option("-S", "--spb", type="int", default=25, help="set samples/baud [default=%default]")
    parser.add_option("-b", "--barker", action="store_true", default=False, help="Use Barker Spreading [default=%default]")
    parser.add_option("-p", "--no-crc-check", action="store_true", default=False, help="Check payload crc [default=%default]")
    parser.add_option("-v", "--verbose", action="store_true", default=False, help="Verbose Output")

    (options, args) = parser.parse_args()
    if len(args) != 0:
      parser.print_help()
      sys.exit(1)

    self.u = usrp_rx(options.decim, options.verbose, options.gain, options.freq)        
    self.bit_receiver = bbn_80211b_demod_pkts(spb=options.spb,
                                                  alpha=0.5,
                                                  use_barker=options.barker,
                                                  check_crc=
                                                  not options.no_crc_check,
                                                  callback = rx_callback)

    self.connect(self.u, self.bit_receiver)


def main ():
  app = app_flow_graph()
  app.start()
  os.read(0,10)

if __name__ == '__main__':
    main ()
