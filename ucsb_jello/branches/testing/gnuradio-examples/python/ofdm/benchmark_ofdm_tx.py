#!/usr/bin/env python
#
# Copyright 2005, 2006 Free Software Foundation, Inc.
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

import time, struct, sys

# from current dir
from transmit_path import transmit_path
from pick_bitrate import pick_tx_bitrate
import fusb_options

class my_top_block(gr.top_block):
    def __init__(self, options):
        gr.top_block.__init__(self)

        self._tx_freq            = options.tx_freq         # tranmitter's center frequency
        self._tx_subdev_spec     = options.tx_subdev_spec  # daughterboard to use
        self._interp             = options.interp          # interpolating rate for the USRP (prelim)
        self._fusb_block_size    = options.fusb_block_size # usb info for USRP
        self._fusb_nblocks       = options.fusb_nblocks    # usb info for USRP
        self._which              = options.which           # linklab, which USRP to use

        if self._tx_freq is None:
            sys.stderr.write("-f FREQ or --freq FREQ or --tx-freq FREQ must be specified\n")
            raise SystemExit

        # Set up USRP sink; also adjusts interp, and bitrate
        self._setup_usrp_sink()

        # copy the final answers back into options for use by modulator
        #options.bitrate = self._bitrate

        self.txpath = transmit_path(options)

        self.connect(self.txpath, self.u)
        
    def _setup_usrp_sink(self):
        """
        Creates a USRP sink, determines the settings for best bitrate,
        and attaches to the transmitter's subdevice.
        """
        # linklab, specify which USRP to use
        self.u = usrp.sink_c(self._which, fusb_block_size=self._fusb_block_size,
                             fusb_nblocks=self._fusb_nblocks)

        self.u.set_interp_rate(self._interp)

        # determine the daughterboard subdevice we're using
        if self._tx_subdev_spec is None:
            self._tx_subdev_spec = usrp.pick_tx_subdevice(self.u)
        self.u.set_mux(usrp.determine_tx_mux_value(self.u, self._tx_subdev_spec))
        self.subdev = usrp.selected_subdev(self.u, self._tx_subdev_spec)

        # Set center frequency of USRP
        ok = self.set_freq(self._tx_freq)
        if not ok:
            print "Failed to set Tx frequency to %s" % (eng_notation.num_to_str(self._tx_freq),)
            raise ValueError

        # Set the USRP for maximum transmit gain
        # (Note that on the RFX cards this is a nop.)
        self.set_gain(self.subdev.gain_range()[1])

        # enable Auto Transmit/Receive switching
        self.set_auto_tr(True)

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
        r = self.u.tune(self.subdev.which(), self.subdev, target_freq)
        if r:
            return True

        return False
        
    def set_gain(self, gain):
        """
        Sets the analog gain in the USRP
        """
        self.gain = gain
        self.subdev.set_gain(gain)

    def set_auto_tr(self, enable):
        """
        Turns on auto transmit/receive of USRP daughterboard (if exits; else ignored)
        """
        return self.subdev.set_auto_tr(enable)
        
    def interp(self):
        return self._interp

    def add_options(normal, expert):
        """
        Adds usrp-specific options to the Options Parser
        """
        add_freq_option(normal)
        normal.add_option("-T", "--tx-subdev-spec", type="subdev", default=None,
                          help="select USRP Tx side A or B")
        normal.add_option("-v", "--verbose", action="store_true", default=False)
        # linklab,  add options to specify which USRP to sue 
        normal.add_option("-w", "--which", type="int", default=0,
                          help="select which USRP (0, 1, ...) default is %default",  metavar="NUM")

        expert.add_option("", "--tx-freq", type="eng_float", default=None,
                          help="set transmit frequency to FREQ [default=%default]", metavar="FREQ")
        expert.add_option("-i", "--interp", type="intx", default=256,
                          help="set fpga interpolation rate to INTERP [default=%default]")
    # Make a static method to call before instantiation
    add_options = staticmethod(add_options)

    def _print_verbage(self):
        """
        Prints information about the transmit path
        """
        print "Using TX d'board %s"    % (self.subdev.side_and_name(),)
        print "modulation:      %s"    % (self._modulator_class.__name__)
        print "interp:          %3d"   % (self._interp)
        print "Tx Frequency:    %s"    % (eng_notation.num_to_str(self._tx_freq))
        

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

    def send_pkt(payload='', eof=False):
        return tb.txpath.send_pkt(payload, eof)

    parser = OptionParser(option_class=eng_option, conflict_handler="resolve")
    expert_grp = parser.add_option_group("Expert")
    parser.add_option("-s", "--size", type="eng_float", default=400,
                      help="set packet size [default=%default]")
    parser.add_option("-M", "--megabytes", type="eng_float", default=1.0,
                      help="set megabytes to transmit [default=%default]")
    parser.add_option("","--discontinuous", action="store_true", default=False,
                      help="enable discontinuous mode")

    my_top_block.add_options(parser, expert_grp)
    transmit_path.add_options(parser, expert_grp)
    blks2.ofdm_mod.add_options(parser, expert_grp)
    blks2.ofdm_demod.add_options(parser, expert_grp)
    fusb_options.add_options(expert_grp)

    (options, args) = parser.parse_args ()

    # build the graph
    tb = my_top_block(options)
    
    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
        print "Warning: failed to enable realtime scheduling"

    tb.start()                       # start flow graph
    
    # generate and send packets
    nbytes = int(1e6 * options.megabytes)
    n = 0
    pktno = 0
    pkt_size = int(options.size)

    while n < nbytes:
        send_pkt(struct.pack('!H', pktno) + (pkt_size - 2) * chr(pktno & 0xff))
        n += pkt_size
        sys.stderr.write('.')
        if options.discontinuous and pktno % 5 == 1:
            time.sleep(1)
        pktno += 1
        
    send_pkt(eof=True)
    tb.wait()                       # wait for it to finish

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
