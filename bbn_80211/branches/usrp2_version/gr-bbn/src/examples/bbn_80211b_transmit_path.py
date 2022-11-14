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
from bbn_80211b_pkt import *

# /////////////////////////////////////////////////////////////////////////////
#                              transmit path
# /////////////////////////////////////////////////////////////////////////////

class bbn_80211b_transmit_path(gr.hier_block2):
  def __init__(self, subdev_spec, interp, spb, use_barker):
    gr.hier_block2.__init__(self, "bbn_80211b_transmit_path", gr.io_signature(0,0,0), gr.io_signature(0,0,0))
    self.normal_gain = 28000

#    self.u = usrp.sink_c()
    self.u = usrp2.sink_32fc("eth1",0)
    dac_rate = self.u.dac_rate();

    self._spb = spb
    self._interp=int(interp)
    self.u.set_interp(self._interp)



    # transmitter
    self.packet_transmitter = bbn_80211b_mod_pkts(spb=spb,
                                                  alpha=0.5,
                                                  gain=self.normal_gain,
                                                  use_barker=use_barker)
    
    self.connect(self.packet_transmitter, self.u)
    self.u.set_gain(self.u.gain_range()[1])  # set max Tx gain
    #self.set_auto_tr(True)    # enable Auto Transmit/Receive switching

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
    r = self.u.set_center_freq(target_freq)
    if r:
      # Could use residual_freq in s/w freq translator
      return True

    return False

  def set_gain(self, gain):
    self.gain = gain
    self.u.set_gain(gain)

  def set_auto_tr(self, enable):
    return self.u.set_auto_tr(enable)
      
  def send_pkt(self, payload='', send_rate=0, eof=False):
    return self.packet_transmitter.send_pkt(payload, send_rate, eof)
      
  def spb(self):
    return self._spb

  def interp(self):
    return self._interp
