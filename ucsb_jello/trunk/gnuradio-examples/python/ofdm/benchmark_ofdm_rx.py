#!/usr/bin/env python
#
# Copyright 2006, 2007 Free Software Foundation, Inc.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from gnuradio import gr, blks2
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser

import struct, sys
# linklab, import math
import math
# from current dir
from receive_path import receive_path
import fusb_options

class my_top_block(gr.top_block):
    def __init__(self, callback, options):
        gr.top_block.__init__(self)

        self._rx_freq            = options.rx_freq         # receiver's center frequency
        self._rx_gain            = options.rx_gain         # receiver's gain
        self._rx_subdev_spec     = options.rx_subdev_spec  # daughterboard to use
        self._decim              = options.decim           # Decimating rate for the USRP (prelim)
        self._fusb_block_size    = options.fusb_block_size # usb info for USRP
        self._fusb_nblocks       = options.fusb_nblocks    # usb info for USRP
        self._which              = options.which           # linklab, which USRP to use

        if self._rx_freq is None:
            sys.stderr.write("-f FREQ or --freq FREQ or --rx-freq FREQ must be specified\n")
            raise SystemExit

        # Set up USRP source
        self._setup_usrp_source()
        ok = self.set_freq(self._rx_freq)
        if not ok:
            print "Failed to set Rx frequency to %s" % (eng_notation.num_to_str(self._rx_freq))
            raise ValueError, eng_notation.num_to_str(self._rx_freq)
        g = self.subdev.gain_range()
        if options.show_rx_gain_range:
            print "Rx Gain Range: minimum = %g, maximum = %g, step size = %g" \
                  % (g[0], g[1], g[2])
        self.set_gain(options.rx_gain)
        self.set_auto_tr(True)                 # enable Auto Transmit/Receive switching

        # Set up receive path
        self.rxpath = receive_path(callback, options)

        self.connect(self.u, self.rxpath)
        
    def _setup_usrp_source(self):
        # linklab, specify which USRP to use
        self.u = usrp.source_c (self._which, fusb_block_size=self._fusb_block_size,
                                fusb_nblocks=self._fusb_nblocks)
        adc_rate = self.u.adc_rate()

        self.u.set_decim_rate(self._decim)

        # determine the daughterboard subdevice we're using
        if self._rx_subdev_spec is None:
            self._rx_subdev_spec = usrp.pick_rx_subdevice(self.u)
        self.subdev = usrp.selected_subdev(self.u, self._rx_subdev_spec)

        self.u.set_mux(usrp.determine_rx_mux_value(self.u, self._rx_subdev_spec))

    def set_freq(self, target_freq):
        """
        Set the center frequency we're interested in.

        @param target_freq: frequency in Hz
        @rypte: bool

        Tuning is a two step process.  First we ask the front-end to
        tune as close to the desired frequency as it can.  Then we use
        the result of that operation and our target_frequency to
        determine the value for the digital up converter.
        """
        r = self.u.tune(0, self.subdev, target_freq)
        if r:
            return True

        return False

    def set_gain(self, gain):
        """
        Sets the analog gain in the USRP
        """
        if gain is None:
            r = self.subdev.gain_range()
            gain = (r[0] + r[1])/2               # set gain to midpoint
        self.gain = gain
        return self.subdev.set_gain(gain)

    def set_auto_tr(self, enable):
        return self.subdev.set_auto_tr(enable)

    def decim(self):
        return self._decim

    def add_options(normal, expert):
        """
        Adds usrp-specific options to the Options Parser
        """
        add_freq_option(normal)
        normal.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
                          help="select USRP Rx side A or B")
        normal.add_option("", "--rx-gain", type="eng_float", default=None, metavar="GAIN",
                          help="set receiver gain in dB [default=midpoint].  See also --show-rx-gain-range")
        normal.add_option("", "--show-rx-gain-range", action="store_true", default=False, 
                          help="print min and max Rx gain available on selected daughterboard")
        normal.add_option("-v", "--verbose", action="store_true", default=False)
        # linklab,  add options to specify which USRP to sue 
        normal.add_option("-w", "--which", type="int", default=0,
                          help="select which USRP (0, 1, ...) default is %default",  metavar="NUM")

        expert.add_option("", "--rx-freq", type="eng_float", default=None,
                          help="set Rx frequency to FREQ [default=%default]", metavar="FREQ")
        expert.add_option("-d", "--decim", type="intx", default=128,
                          help="set fpga decimation rate to DECIM [default=%default]")
        expert.add_option("", "--snr", type="eng_float", default=30,
                          help="set the SNR of the channel in dB [default=%default]")
   

    # Make a static method to call before instantiation
    add_options = staticmethod(add_options)

def add_freq_option(parser):
    """
    Hackery that has the -f / --freq option set both tx_freq and rx_freq
    """
    def freq_callback(option, opt_str, value, parser):
        parser.values.rx_freq = value
        parser.values.tx_freq = value

    if not parser.has_option('--freq'):
        parser.add_option('-f', '--freq', type="eng_float",
                          action="callback", callback=freq_callback,
                          help="set Tx and/or Rx frequency to FREQ [default=%default]",
                          metavar="FREQ")

# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():

    global n_rcvd, n_right
        
    n_rcvd = 0
    n_right = 0

    # linklab, add parameters for interger freq offset (int_fo), fractional freq offset (frac_fo)
    # SINR estimation in time domain (time_sinr), SINR estimation in freq domain
    def rx_callback(ok, payload, int_fo, frac_fo, time_sinr, freq_sinr):
        global n_rcvd, n_right
        n_rcvd += 1
        try:
            (pktno,) = struct.unpack('!H', payload[0:2])
        except:
            pktno = 1
        if ok:
            n_right += 1
            freq_offset = int_fo+frac_fo/math.pi
            print "freq offset: %+.2f(subcarriers) \t SINR: %.2f(time domain), %.2f(freq domain)" % (freq_offset, time_sinr, freq_sinr)
        # linklab, calculate packet loss rate and error rate
        try :
            pkt_loss = (1-n_rcvd/float(pktno))*100
            pkt_err = (1-n_right/float(pktno))*100
        except:
            pkt_loss = 100
            pkt_err = 100
        
        # linklab, print packet info
        print "ok: %r \t pktno: %d \t n_rcvd: %d \t n_right: %d \t pkt_loss: %.2f%% \t pkt_err: %.2f%%" % (ok, pktno, n_rcvd, n_right, pkt_loss, pkt_err)
        
        if 0:
            printlst = list()
            for x in payload[2:]:
                t = hex(ord(x)).replace('0x', '')
                if(len(t) == 1):
                    t = '0' + t
                printlst.append(t)
            printable = ''.join(printlst)

            print printable
            print "\n"

    parser = OptionParser(option_class=eng_option, conflict_handler="resolve")
    expert_grp = parser.add_option_group("Expert")
    parser.add_option("","--discontinuous", action="store_true", default=False,
                      help="enable discontinuous")

    my_top_block.add_options(parser, expert_grp)
    receive_path.add_options(parser, expert_grp)
    blks2.ofdm_mod.add_options(parser, expert_grp)
    blks2.ofdm_demod.add_options(parser, expert_grp)
    fusb_options.add_options(expert_grp)

    (options, args) = parser.parse_args ()

    # build the graph
    tb = my_top_block(rx_callback, options)

    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
        print "Warning: failed to enable realtime scheduling"

    tb.start()                      # start flow graph
    tb.wait()                       # wait for it to finish

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
