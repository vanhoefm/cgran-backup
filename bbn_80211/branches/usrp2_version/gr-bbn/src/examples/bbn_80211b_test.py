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

from gnuradio import gr, blks2
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
  print "Payload: ", packet_data

class my_block(gr.top_block):

  def __init__(self, rx_callback, spb, alpha, SNR, use_barker):
    # m is constellation size
    # if diff==True we are doing DxPSK
    gr.top_block.__init__(self)

    tb = self

    # transmitter
    self.packet_transmitter = bbn_80211b_mod_pkts(spb=spb, alpha=alpha, gain=1, use_barker=use_barker)

    # add some noise
    add = gr.add_cc()
    noise = gr.noise_source_c(gr.GR_GAUSSIAN, pow(10.0,-SNR/20.0))

    # channel filter
    length = 4
    rx_filt_taps = gr.firdes.low_pass(1,length,0.8,0.1,gr.firdes.WIN_HANN)
    self.rx_filt = gr.fir_filter_ccf(1,rx_filt_taps)

 # receiver
    self.bit_receiver = bbn_80211b_demod_pkts(spb=spb, alpha=alpha, callback=rx_callback, use_barker=use_barker)
    #tb.connect(self.packet_transmitter, self.bit_receiver)
    tb.connect(noise, (add,0))
    tb.connect(self.packet_transmitter, self.rx_filt, (add, 1))
    tb.connect(add, self.bit_receiver)
    
    

class stats(object):
  def __init__(self):
    self.npkts = 0
    self.nright = 0
        
def main():
  st = stats()
  
  def send_pkt(payload='', send_rate=0, eof=False):
      tb.packet_transmitter.send_pkt(payload, send_rate, eof)

  parser = OptionParser (option_class=eng_option)
  parser.add_option("","--spb", type=int, default=25,
                    help="set samples per baud to SPB [default=%default]")
  parser.add_option("", "--alpha", type="eng_float", default=0.5,
                    help="set excess bandwidth for RRC filter [default=%default]")
  parser.add_option("", "--snr", type="eng_float", default=12,
                    help="set SNR in dB for simulation [default=%default]")
  parser.add_option("", "--barker", type=int, default=1, help="use barker spreading [default=%default]")

  (options, args) = parser.parse_args ()

  if len(args) != 0:
    parser.print_help()
    sys.exit(1)


  print "SNR set at: ", options.snr, " spb set at: ", options.spb, " using barker: ", options.barker
  tb = my_block(rx_callback, options.spb, options.alpha, options.snr, options.barker)

  tb.start()
  
  
  n = 0
  pktno = 0
  #send_pkt('The quick brown fox jumps over the lazy dog.')
  i = 0
  while i < 100:
    send_pkt('012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789 pkt:' + str(i),1)
    i+=1
  #i = 0
  #while i < 1000000:
  #  i = i + 1
  send_pkt('01234567890123456789',2)
  
  send_pkt('01234567890123456789',2)
  
  send_pkt('01234567890123456789',2)
  
  send_pkt('The quick brown fox jumps over the lazy dog.', 2)
  
  send_pkt('The quick brown fox jumps over the lazy dog.', 1)
  
  
  
  send_pkt(eof=True) # tell modulator we're not sending any more pkts

  tb.wait()


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    pass
