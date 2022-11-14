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
from bbn_80211b_pkt import *

# /////////////////////////////////////////////////////////////////////////////
#                              transmit path
# /////////////////////////////////////////////////////////////////////////////

class bbn_80211b_transmit_path(gr.hier_block):
    def __init__(self, fg, subdev_spec, interp, spb, use_barker):

        self.normal_gain = 28000

        self.u = usrp.sink_c()
        dac_rate = self.u.dac_rate();

        self._spb = spb
        self._interp=int(interp)
        self.u.set_interp_rate(self._interp)

        # determine the daughterboard subdevice we're using
        if subdev_spec is None:
            subdev_spec = usrp.pick_tx_subdevice(self.u)

        self.u.set_mux(usrp.determine_tx_mux_value(self.u, subdev_spec))
        self.subdev = usrp.selected_subdev(self.u, subdev_spec)
        print "Using TX d'board %s" % (self.subdev.side_and_name(),)

        # transmitter
        self.packet_transmitter = bbn_80211b_mod_pkts(fg, spb=spb,
                                                      alpha=0.5,
                                                      gain=self.normal_gain,
                                                      use_barker=use_barker)
        fg.connect(self.packet_transmitter, self.u)
        gr.hier_block.__init__(self, fg, None, None)

        self.set_gain(self.subdev.gain_range()[1])  # set max Tx gain
        self.set_auto_tr(True)    # enable Auto Transmit/Receive switching

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
        r = self.u.tune(self.subdev._which, self.subdev, target_freq)
        if r:
            # Could use residual_freq in s/w freq translator
            return True

        return False

    def set_gain(self, gain):
        self.gain = gain
        self.subdev.set_gain(gain)

    def set_auto_tr(self, enable):
        return self.subdev.set_auto_tr(enable)
        
    def send_pkt(self, payload='', eof=False):
        return self.packet_transmitter.send_pkt(payload, eof)
        
    def spb(self):
        return self._spb

    def interp(self):
        return self._interp
