#!/usr/bin/env python
#
# Copyright 2005,2006 Free Software Foundation, Inc.
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


# /////////////////////////////////////////////////////////////////////////////
#
#    This code sets up up a virtual ethernet interface (typically gr0),
#    and relays packets between the interface and the GNU Radio PHY+MAC
#
#    What this means in plain language, is that if you've got a couple
#    of USRPs on different machines, and if you run this code on those
#    machines, you can talk between them using normal TCP/IP networking.
#
# /////////////////////////////////////////////////////////////////////////////


from gnuradio import gr, gru, blks
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser

import random
import time
import struct
import sys
import os
import platform
import socket

# from current dir
from transmit_path import transmit_path
from receive_path import receive_path
from mac_utils import *
from framing import *
import fusb_options

# /////////////////////////////////////////////////////////////////////////////
#
#   Use the Universal TUN/TAP device driver to move packets to/from kernel
#
#   See /usr/src/linux/Documentation/networking/tuntap.txt
#
# /////////////////////////////////////////////////////////////////////////////

# Linux specific...
# TUNSETIFF ifr flags from <linux/tun_if.h>

IFF_TUN		= 0x0001   # tunnel IP packets
IFF_TAP		= 0x0002   # tunnel ethernet frames
IFF_NO_PI	= 0x1000   # don't pass extra packet info
IFF_ONE_QUEUE	= 0x2000   # beats me ;)

def open_tun_interface(type, tun_device_filename):
    from fcntl import ioctl

    mode = IFF_TUN   
    ifname = 'tun0'

    if type == 'tap':
       mode = IFF_TAP | IFF_NO_PI
       ifname = 'tapxx'

    TUNSETIFF = 0x400454ca
    TAPGIFNAME = 0x40206500
    tun = os.open(tun_device_filename, os.O_RDWR)
    if platform.system() == 'Linux':
	ifs = ioctl(tun, TUNSETIFF, struct.pack("16sH", "gr%d", mode))
    	ifname = ifs[:16].strip("\x00")
    if platform.system() == 'NetBSD':
	ifs = ioctl(tun, TAPGIFNAME, struct.pack("16sH", "gr%d", mode))
	ifname = ifs[:16].strip("\x00")
    return (tun, ifname)
	

# /////////////////////////////////////////////////////////////////////////////
#                             the flow graph
# /////////////////////////////////////////////////////////////////////////////

class my_graph(gr.flow_graph):

    def __init__(self, mod_class, demod_class,
                 tx_subdev_spec, rx_subdev_spec,
                 rx_callback,
                 options, kwargs):

        gr.flow_graph.__init__(self)
        self.txpath = transmit_path(self, mod_class, tx_subdev_spec,
                                    options.bitrate, options.interp, options.spb,
                                    options.tx_gain, options, kwargs)
        self.rxpath = receive_path(self, demod_class, rx_subdev_spec,
                                   options.bitrate, options.decim, options.spb,
                                   rx_callback, options, {})

    def send_pkt(self, payload='', eof=False):
        return self.txpath.send_pkt(payload, eof)

    def carrier_sensed(self):
        """
        Return True if the receive path thinks there's carrier
        """
        return self.rxpath.carrier_sensed()


# /////////////////////////////////////////////////////////////////////////////
#                           Carrier Sense MAC
# /////////////////////////////////////////////////////////////////////////////

class cs_mac(object):
    """
    Prototype carrier sense MAC

    Reads packets from the TUN/TAP interface, and sends them to the PHY.
    Receives packets from the PHY via phy_rx_callback, and sends them
    into the TUN/TAP interface.

    Of course, we're not restricted to getting packets via TUN/TAP, this
    is just an example.
    """
    def __init__(self, tun_fd, mac_addr, pkttype, bssid, cache, verbose=False):
        self.tun_fd = tun_fd       # file descriptor for TUN/TAP interface
        self.verbose = verbose
        self.fg = None             # flow graph (access to PHY)
        self.mac_addr = mac_addr
	self.pkttype = pkttype
        self.bssid = bssid
        self.cache = cache
	self.receiver = packet_receiver(mac_addr, bssid, \
                        self.mac_rcv_callback, verbose)

    def set_flow_graph(self, fg):
        self.fg = fg

    def mac_rcv_callback(self, packet):
        """ 
        Invoked by packet_receiver.receive when it has a complete
        packet to pass up to the OS.
        """
        if self.verbose:
            print "Pkt Rx: pkt len = %4d" % (len(packet))

        os.write(self.tun_fd, packet)
        

    def phy_rx_callback(self, ok, payload):
        """
        Invoked by thread associated with PHY to pass received packet up.

        @param ok: bool indicating whether payload CRC was OK
        @param payload: contents of the packet (string)
        """
        if self.verbose:
            print "Rx: ok = %r  len(payload) = %4d" % (ok, len(payload))
        if ok:
            self.receiver.receive(payload)

    def main_loop(self):
        """
        Main loop for MAC.
        Only returns if we get an error reading from TUN.

        FIXME: may want to check for EINTR and EAGAIN and reissue read
        """
        min_delay = 0.001               # seconds

	sender = packet_sender(self.pkttype, self.mac_addr, self.bssid, \
                               self.fg.send_pkt, self.cache, self.verbose)

        while 1:
            payload = os.read(self.tun_fd, 10*1024)
            if not payload:
                self.fg.send_pkt(eof=True)
                break

            if self.verbose:
                print "Tx: len(payload) = %4d" % (len(payload),)

            delay = min_delay
            while self.fg.carrier_sensed():
                sys.stderr.write('B')
                time.sleep(delay)
                if delay < 0.050:
                    delay = delay * 2       # exponential back-off

            sender.send(payload)


# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():

    tun_device_default = "/dev/net/tun"
    if (platform.system() == 'FreeBSD' or platform.system() == 'NetBSD'):
       tun_device_default = "/dev/tap"

    parser = OptionParser (option_class=eng_option, 
	                   usage="usage: %prog [options] [ipaddr=macaddr] ...")
    parser.add_option("-f", "--freq", type="eng_float", default=423.1e6,
                       help="set Tx and Rx frequency to FREQ [default=%default]", metavar="FREQ")
    parser.add_option("-r", "--bitrate", type="eng_float", default=None,
                      help="specify bitrate.  spb and interp will be derived.")
    parser.add_option("-g", "--rx-gain", type="eng_float", default=27,
                      help="set rx gain")
    parser.add_option("-p", "--tx-gain", type="eng_float", default=100,
                      help="set tx gain")
    parser.add_option("-T", "--tx-subdev-spec", type="subdev", default=None,
                      help="select USRP Tx side A or B")
    parser.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
                      help="select USRP Rx side A or B")
    parser.add_option("-S", "--spb", type="int", default=None, help="set samples/baud [default=%default]")
    parser.add_option("-d", "--decim", type="intx", default=None,
                      help="set fpga decim rate to DECIM [default=%default]")
    parser.add_option("-i", "--interp", type="intx", default=None,
                      help="set fpga interpolation rate to INTERP [default=%default]")
    parser.add_option("-c", "--carrier-threshold", type="eng_float", default=30,
                      help="set carrier detect threshold (dB) [default=%default]")
    parser.add_option("", "--bt", type="float", default=0.3, help="set bandwidth-time product [default=%default]")
    parser.add_option("","--tun-device-filename", default=tun_device_default,
                      help="path to tun device file [default=%default]")
    parser.add_option("-b","--bssid", default="00:00:00:00:00:00", 
		    help="set bssid for network in the form xx:xx:xx:xx:xx:xx")
    parser.add_option("","--tun", action="store_true", default=False, 
		    help="use tun device instead of tap to pass packets.")
    parser.add_option("","--macfile", default="simple-config", 
                    help="filename containing a set of ipaddr=macaddr mappings, only needed with --tun.")
    parser.add_option("-v","--verbose", action="store_true", default=False)
    fusb_options.add_options(parser)
    (options, args) = parser.parse_args ()

    bssid = validate_mac_addr(options.bssid)
    if bssid == 0:
	print "Invalid BSSID ", options.bssid
        parser.print_help()
        sys.exit(1)

    if options.freq < 1e6:
        options.freq *= 1e6

    mod_kwargs = {
        'bt' : options.bt,
        }

    pkttype = 'eth'
    tuntype = 'tap'
    mcache = None
    if options.tun:
        pkttype = 'ip'
        tuntype = 'tun'
   
        # If a file of IP address to MAC address mappings has been provided,
        # open it and read it in.  Since the format is the same as the command
        # line mappings, add it to the end of the args list so they can both
        # be processed at the same time
        if options.macfile != None:
            try: 
      	        config = file(options.macfile, 'r')
                for line in config:
                    line = line.strip()
                    # skip blank lines
                    if line == '':
                       continue
                    # skip comment lines
                    if line[0] == '#':
                       continue
                    args.append(line)
                config.close()
            except IOError:
                print "Cannot open configuration file: ", options.macfile

        # remove the mac addr.  
        # The remaining arguments should be ip->mac mappings
	del args[0]
        mcache = mac_cache()
        for map in args:
            addrs = map.split('=', 1)
            if len(addrs) != 2:
               print "Invalid ipaddr=macaddr mapping ", map
               parser.print_help()
               sys.exit(1)
            try: 
               ipaddr = socket.inet_aton(addrs[0])
            except socket.error:
      	       print "Invalid IP addr ", addrs[0]
               parser.print_help()
               sys.exit(1)
            macaddr = validate_mac_addr(addrs[1])
            if macaddr == 0:
	       print "Invalid MAC addr ", addrs[1]
               parser.print_help()
               sys.exit(1)
            mcache.add(ipaddr, macaddr)
        if options.verbose:
            print "Configured MAC addresses:"
            mcache.print_map()
    # end if options.tun
        

    # open the TUN/TAP interface
    (tun_fd, tun_ifname) = open_tun_interface(tuntype, options.tun_device_filename)
    tun_mac = get_mac_for_interface(tun_ifname)
    mac_addr = validate_mac_addr(tun_mac)
    if mac_addr == 0:
        print "Invalid MAC address ", tun_mac, " for interface ", tun_ifname
	print "exiting."
        sys.exit(1)
    if options.verbose:
        print "Using MAC address ", tun_mac, " for interface ", tun_ifname


    # Attempt to enable realtime scheduling
    r = gr.enable_realtime_scheduling()
    if r == gr.RT_OK:
        realtime = True
    else:
        realtime = False
        print "Note: failed to enable realtime scheduling"


    # If the user hasn't set the fusb_* parameters on the command line,
    # pick some values that will reduce latency.

    if options.fusb_block_size == 0 and options.fusb_nblocks == 0:
        if realtime:                        # be more aggressive
            options.fusb_block_size = gr.prefs().get_long('fusb', 'rt_block_size', 1024)
            options.fusb_nblocks    = gr.prefs().get_long('fusb', 'rt_nblocks', 16)
        else:
            options.fusb_block_size = gr.prefs().get_long('fusb', 'block_size', 4096)
            options.fusb_nblocks    = gr.prefs().get_long('fusb', 'nblocks', 16)
    
    print "fusb_block_size =", options.fusb_block_size
    print "fusb_nblocks    =", options.fusb_nblocks

    # instantiate the MAC
    mac = cs_mac(tun_fd, mac_addr, pkttype, bssid, mcache, verbose=True)


    # build the graph (PHY)
    fg = my_graph(blks.gmsk2_mod, blks.gmsk2_demod,
                  options.tx_subdev_spec, options.rx_subdev_spec,
                  mac.phy_rx_callback, 
                  options, mod_kwargs)

    mac.set_flow_graph(fg)    # give the MAC a handle for the PHY

    if fg.txpath.bitrate() != fg.rxpath.bitrate():
        print "WARNING: Transmit bitrate = %sb/sec, Receive bitrate = %sb/sec" % (
            eng_notation.num_to_str(fg.txpath.bitrate()),
            eng_notation.num_to_str(fg.rxpath.bitrate()))
             
    print "bitrate: %sb/sec" % (eng_notation.num_to_str(fg.txpath.bitrate()),)
    print "spb:     %3d" % (fg.txpath.spb(),)
    print "interp:  %3d" % (fg.txpath.interp(),)

    ok = fg.txpath.set_freq(options.freq)
    if not ok:
        print "Failed to set Tx frequency to %s" % (eng_notation.num_to_str(options.freq),)
        raise SystemExit

    ok = fg.rxpath.set_freq(options.freq)
    if not ok:
        print "Failed to set Rx frequency to %s" % (eng_notation.num_to_str(options.freq),)
        raise SystemExit

    fg.rxpath.set_gain(options.rx_gain)
    print "Rx gain_range: ", fg.rxpath.subdev.gain_range(), " using", fg.rxpath.gain

    fg.rxpath.set_carrier_threshold(options.carrier_threshold)
    print "Carrier sense threshold:", options.carrier_threshold, "dB"
    
    print
    print "Allocated virtual ethernet interface: %s" % (tun_ifname,)
    print "You must now use ifconfig to set its IP address. E.g.,"
    print
    print "  $ sudo ifconfig %s 10.10.10.1" % (tun_ifname,)
    print
    print "Be sure to use a different address in the same subnet for each machine."
    print


    fg.start()    # Start executing the flow graph (runs in separate threads)

    mac.main_loop()    # don't expect this to return...

    fg.stop()     # but if it does, tell flow graph to stop.
    fg.wait()     # wait for it to finish
                

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
