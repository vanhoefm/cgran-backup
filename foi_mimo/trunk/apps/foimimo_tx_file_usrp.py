#!/usr/bin/env python
#
# Copyright 2011 FOI
# 
# This file is part of FOI-MIMO
# 
# FOI-MIMO is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# FOI-MIMO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FOI-MIMO; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from foimimo import *
from foimimo.foi_mimo_tx_path import mimo_transmit_path as mimo_tx_path
from foimimo.foi_siso_tx_path import siso_transmit_path as siso_tx_path
from gui.foimimo_top_blocks import *

from gnuradio import gr, blks2
from gnuradio import uhd
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from gnuradio.gr import firdes
from grc_gnuradio import blks2 as grc_blks2
from optparse import OptionParser
import random, time, struct, sys, math, os

VERBOSE = 0

class top_block_foimimo(gr.top_block):

    def __init__(self, options):
        gr.top_block.__init__(self)

        ##################################################
        # Variables
        ##################################################

        symbols_per_packet = math.ceil(((4+options.size+4) * 8) / options.occupied_tones)
        samples_per_packet = (symbols_per_packet+2) * (options.fft_length+options.cp_length)
        print "Symbols per Packet: ", symbols_per_packet
        print "Samples per Packet: ", samples_per_packet
        if options.discontinuous:
            stream_size = [100000, int(options.discontinuous*samples_per_packet)]
        else:
            stream_size = [0, 100000]
            

        ##################################################
        # Blocks
        ##################################################
        if options.siso:
            device_addr = "addr0="+options.usrp_addr0
            print "Using USRP unit with device adress: ",device_addr
            self.uhd_usrp_sink = uhd.single_usrp_sink(
                device_addr=device_addr,
                io_type=uhd.io_type.COMPLEX_FLOAT32,
                num_channels=1,
                )
            # Set usrp sink parameters
            self.uhd_usrp_sink.set_clock_config(uhd.clock_config.external());
            self.uhd_usrp_sink.set_time_next_pps(uhd.time_spec())
            self.uhd_usrp_sink.set_samp_rate(options.sample_rate)
            self.uhd_usrp_sink.set_center_freq(options.center_freq, 0)
            self.uhd_usrp_sink.set_gain(options.gain_tx, 0)

            self.foi_txpath = siso_tx_path(options)

        else:
            device_addr = "addr0="+options.usrp_addr0+", addr1="+options.usrp_addr1
            print "Using USRP units with device adress: ",device_addr
            self.uhd_usrp_sink = uhd.multi_usrp_sink(
                device_addr=device_addr,
                io_type=uhd.io_type.COMPLEX_FLOAT32,
                num_channels=2,
                )
            # OBS Set usrp sink parameters
            self.uhd_usrp_sink.set_clock_config(uhd.clock_config.external(), uhd.ALL_MBOARDS);
            self.uhd_usrp_sink.set_time_unknown_pps(uhd.time_spec())
            self.uhd_usrp_sink.set_samp_rate(options.sample_rate)
            self.uhd_usrp_sink.set_center_freq(options.center_freq, 0)
            self.uhd_usrp_sink.set_center_freq(options.center_freq, 1)
            self.uhd_usrp_sink.set_gain(options.gain_tx, 0)
            self.uhd_usrp_sink.set_gain(options.gain_tx, 1)

            self.foi_txpath = mimo_tx_path(options)

    
        ##################################################
        # Connections
        ##################################################
        if options.siso:
            self.connect((self.foi_txpath, 0), (self.uhd_usrp_sink, 0))
        else:
            self.connect((self.foi_txpath, 0), (self.uhd_usrp_sink, 0))
            self.connect((self.foi_txpath, 1), (self.uhd_usrp_sink, 1))

# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():
    global n_rcvd, n_right
        
    def send_pkt(payload='', eof=False):
        return tb.foi_txpath.send_pkt(payload, eof)
       
    parser = OptionParser(option_class=eng_option, conflict_handler="resolve")
    expert_grp = parser.add_option_group("Expert")
    parser.add_option("-z", "--size", type="eng_float", default=401,
                      help="set packet size [default=%default]")
    parser.add_option("-s","--siso", action="store_true", default=False,
                      help="Enable SISO mode [default is MIMO]")
    parser.add_option("-b","--nbytes", type="eng_float", default=1000000,
                     help="nr of bytes to transmit in BER-mode [default=%default]")
    parser.add_option("-g", "--gain-tx", type="eng_float", default=9,
                      help="Set usrp gain at tx [default=%default]")
    parser.add_option("-u", "--usrp-addr0", type="string", default="192.168.20.2",
                      help="usrp addr, first unit [default=%default]")
    parser.add_option("-w", "--usrp-addr1", type="string", default="192.168.30.2",
                      help="usrp addr, second unit [default=%default]")
    parser.add_option("-f", "--center-freq", type="eng_float", default=433000000.0,
                      help="usrp center frequency [default=%default]")    
    parser.add_option("-r", "--sample-rate", type="eng_float", default=3125000.0,
                      help="sample rate for usrp in Sps [default=%default]") 
    parser.add_option("-i", "--input-filename", type="string", default="",
                      help="input data file, none=BER-mode") 
    parser.add_option("","--discontinuous", type="int", default=0,
                      help="enable discontinous transmission, burst of N packets [Default is continuous]")
    
    siso_tx_path.add_options(parser, expert_grp)
    ofdm_mod.add_options(parser, expert_grp)
    ofdm_mod_with_coding.add_options(parser, expert_grp)
    mimo_tx_path.add_options(parser, expert_grp)
    ofdm_mimo_mod.add_options(parser, expert_grp)
    ofdm_mimo_mod_with_coding.add_options(parser, expert_grp)
    
    (options, args) = parser.parse_args ()
       
    # build the graph
    tb = top_block_tx_gui(options)
    
    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
            print "Warning: failed to enable realtime scheduling"
    
    tb.start()                       # start flow graph

    if options.input_filename == "":
        # generate and send packets
        nbytes = options.nbytes
        n = 0
        s_pktno = 0
        pkt_size = int(options.size)-8  
        while n < nbytes:
            pkt_contents = struct.pack('!H', s_pktno) + (pkt_size - 2) * chr(0 & 0xff)
            send_pkt(pkt_contents)
            sys.stdout.write(".")
            n += pkt_size
            s_pktno += 1
        send_pkt(eof=True)
        print "\nThe transmitter program sent: ", s_pktno-1, "OFDM packets"  
    else:
        # Send packets from infile
        pkt_size = int(options.size)-8
        n_sent = 0
        # Read input data from file:
        in_file = open(options.input_filename,mode="rb")
        indata = in_file.read(pkt_size)
        read_pkt_size = len(indata)
        if VERBOSE:
            print "TB: Reading first pkt of size",read_pkt_size
        while (indata):
            send_pkt(indata)
            n_sent += 1
            if VERBOSE:
                print "TB: Sent pkt!"
            else:
                sys.stdout.write(".")
            indata = in_file.read(pkt_size)
            read_pkt_size = len(indata)
            if VERBOSE:
                print "TB: Reading another pkt of size",read_pkt_size
        else:
            send_pkt(indata)
            send_pkt(eof=True)
            if VERBOSE:
                print "Sent the last empty packet and eof"
            else:
                sys.stdout.write("!")
        in_file.close()
        print "Sent",n_sent,"packets."

    tb.wait()                       # wait for it to finish


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

