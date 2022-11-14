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
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 


from gnuradio import gr, gru, blks2, window
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser

# add timer support
from threading import Timer

from subprocess import *

import random
import time
import struct
import sys
import os
import string
# from current dir
from transmit_path import transmit_path
from receive_path import receive_path
import fusb_options

from sensing_path import sensing_path
from allocation import *
import math
import numpy

n2s = eng_notation.num_to_str

# linklab, define constant
REQUEST_TIMEOUT   = 0.5 # second
MAX_REQ_RETRY     = 5   # times
SYNC_TIMEOUT      = 3   # second
BOFF_TIMEOUT      = 1   # second

# define min sub-carrier num that can sustain a link
MIN_SUBC          = 24  


# linklab states for sender includes
# INIT, TRAN, SENS, COOR, SYNC

REQ_PKT  = 111
ACK_PKT  = 222
SYNC_PKT = 333
DATA_PKT = 444

# carrier map for SYNC state
SYNC_MAP = "fffffff00000000000000000000000000000000000000000000000000000"

# linklab define packet size for data, request, ack and sync
DATA_PKT_SIZE  = 100
ACK_PKT_SIZE   = 100
REQ_PKT_SIZE   = 100

# traffic path with traffic trace files
TRAFFIC_PATH = "./traffic"

# linklab, print time
def print_time():
    print "%8.3f" % (time.time()),
    return

# get frequency offset
def get_freq_offset():

    # check USRP ID
    cmd = "lsusrp -w 0"
    [output, error] = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    if error:
        print "USRP check failed"
        print error
        sys.exit(1)
    else:
        usrp_id = output.split()[4]
        print "USRP check succeedded at", usrp_id

    # open frequency offset file
    file_fo = open("./freq_offset.dat")
    lines = file_fo.readlines()

    offset = 0
    for i in range(len(lines)):
        if usrp_id == lines[i].split()[0]:
            offset = string.atof(lines[i].split()[1])*1000
    return offset

# /////////////////////////////////////////////////////////////////////////////
#                             the flow graph
# /////////////////////////////////////////////////////////////////////////////

class usrp_graph(gr.top_block):
    def __init__(self, callback, options):
        gr.top_block.__init__(self)


        self._tx_freq            = options.tx_freq         # tranmitter's center frequency
        self._tx_subdev_spec     = options.tx_subdev_spec  # daughterboard to use
        self._interp             = options.interp          # interpolating rate for the USRP (prelim)
        self._rx_freq            = options.rx_freq         # receiver's center frequency
        self._rx_gain            = options.rx_gain         # receiver's gain
        self._rx_subdev_spec     = options.rx_subdev_spec  # daughterboard to use
        self._decim              = options.decim           # Decimating rate for the USRP (prelim)
        self._fusb_block_size    = options.fusb_block_size # usb info for USRP
        self._fusb_nblocks       = options.fusb_nblocks    # usb info for USRP
        
        # linklab
        self.carrier_map         = options.carrier_map     # carrier map
        self.occupied_tones      = options.occupied_tones  # occupied tones
        self.which               = options.which           # usrp in use 
        self.id                  = options.id              # link ID
        self.nosense             = options.nosense         # sensing or not
        self.tx_amplitude        = options.tx_amplitude    # tx amplitude
        self._fft_length         = options.fft_length      # fft length
        self._strategy           = options.strategy         # spectrum access strategy
        
        if self._tx_freq is None:
            sys.stderr.write("-f FREQ or --freq FREQ or --tx-freq FREQ must be specified\n")
            raise SystemExit

        if self._rx_freq is None:
            sys.stderr.write("-f FREQ or --freq FREQ or --rx-freq FREQ must be specified\n")
            raise SystemExit

        # Set up USRP sink and source
        self._setup_usrp_sink()
        ok = self.set_snk_freq(self._tx_freq)
        if not ok:
            print "Failed to set Tx frequency to %s" % (eng_notation.num_to_str(self._tx_freq),)
            raise ValueError

        self._setup_usrp_source()
        ok = self.set_src_freq(self._tx_freq)
        if not ok:
            print "Failed to set Rx frequency to %s" % (eng_notation.num_to_str(self._tx_freq),)
            raise ValueError

        # copy the final answers back into options for use by modulator
        #options.bitrate = self._bitrate

        self.txpath = transmit_path(options)
        self.rxpath = receive_path(callback, options)
        self.senspath = sensing_path(options)

        self.connect(self.txpath, self.u_snk)
        self.connect(self.u_src, self.rxpath)
        
        #if options.sender and not self.nosense:
        self.connect(self.u_src, self.senspath)
       

    def carrier_sensed(self):
        """
        Return True if the receive path thinks there's carrier
        """
        return self.rxpath.carrier_sensed()

    def _setup_usrp_sink(self):
        """
        Creates a USRP sink, determines the settings for best bitrate,
        and attaches to the transmitter's subdevice.
        """
        self.u_snk = usrp.sink_c(self.which, fusb_block_size=self._fusb_block_size,
                                 fusb_nblocks=self._fusb_nblocks)

        self.u_snk.set_interp_rate(self._interp)

        # determine the daughterboard subdevice we're using
        if self._tx_subdev_spec is None:
            self._tx_subdev_spec = usrp.pick_tx_subdevice(self.u_snk)
        self.u_snk.set_mux(usrp.determine_tx_mux_value(self.u_snk, self._tx_subdev_spec))
        self.subdev = usrp.selected_subdev(self.u_snk, self._tx_subdev_spec)
        self.subdev_snk = self.subdev

        # Set the USRP for maximum transmit gain
        # (Note that on the RFX cards this is a nop.)
        self.set_gain(self.subdev.gain_range()[1])

        # enable Auto Transmit/Receive switching
        self.set_auto_tr(True)

    def _setup_usrp_source(self):
        self.u_src = usrp.source_c (self.which, fusb_block_size=self._fusb_block_size,
                                fusb_nblocks=self._fusb_nblocks)
 
        adc_rate = self.u_src.adc_rate()

        self.u_src.set_decim_rate(self._decim)

        # determine the daughterboard subdevice we're using
        if self._rx_subdev_spec is None:
            self._rx_subdev_spec = usrp.pick_rx_subdevice(self.u_src)
        self.subdev = usrp.selected_subdev(self.u_src, self._rx_subdev_spec)
        self.subdev_src = self.subdev

        self.u_src.set_mux(usrp.determine_rx_mux_value(self.u_src, self._rx_subdev_spec))

    
    def set_snk_freq(self, target_freq):
        r_snk = self.u_snk.tune(self.subdev_snk.which(), self.subdev_snk, target_freq)
        if r_snk:
            """
            print "Sender @ subdev = ", self.subdev.which()
            print "Baseband frequency is", n2s(r_snk.baseband_freq), "Hz"
            print "DXC frequency is", n2s(r_snk.dxc_freq), "Hz"
            print "Center frequency is", n2s(target_freq), "Hz"
            print "Residual frequency is", n2s(r_snk.residual_freq), "Hz"
            """
            return True

        return False
     
    def set_src_freq(self, target_freq):
        r_src = self.u_src.tune(self.subdev_src.which(), self.subdev_src, target_freq)
        if r_src:
            """
            print "Receiver @ subdev = ", self.subdev.which()
            print "Baseband frequency is", n2s(r_src.baseband_freq), "Hz"
            print "DXC frequency is", n2s(r_src.dxc_freq), "Hz"
            print "Center frequency is", n2s(target_freq), "Hz"
            print "Residual frequency is", n2s(r_src.residual_freq), "Hz"
            """
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
        
    def get_para(self):
        return self._rx_freq, self._fft_length, self._decim

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

        expert.add_option("", "--tx-freq", type="eng_float", default=None,
                          help="set transmit frequency to FREQ [default=%default]", metavar="FREQ")
        expert.add_option("-i", "--interp", type="intx", default=256,
                          help="set fpga interpolation rate to INTERP [default=%default]")
        normal.add_option("-R", "--rx-subdev-spec", type="subdev", default=None,
                          help="select USRP Rx side A or B")
        normal.add_option("", "--rx-gain", type="eng_float", default=None, metavar="GAIN",
                          help="set receiver gain in dB [default=midpoint].  See also --show-rx-gain-range")
        normal.add_option("", "--show-rx-gain-range", action="store_true", default=False, 
                          help="print min and max Rx gain available on selected daughterboard")
        normal.add_option("-v", "--verbose", action="store_true", default=False)


        # linklab
        normal.add_option("-w", "--which", type="int", default=0,
                          help="select which USRP (0, 1, ...) default is %default",  metavar="NUM")
        normal.add_option("", "--id", type="int", default=1,
                          help="select which link (1, 2, ...) the USRP belongs to, default is %default",  metavar="ID")
        normal.add_option("", "--nosense", action="store_true", default=False)


        expert.add_option("", "--rx-freq", type="eng_float", default=None,
                          help="set Rx frequency to FREQ [default=%default]", metavar="FREQ")
        expert.add_option("-d", "--decim", type="intx", default=128,
                          help="set fpga decimation rate to DECIM [default=%default]")
        expert.add_option("", "--snr", type="eng_float", default=30,
                          help="set the SNR of the channel in dB [default=%default]")
        expert.add_option("", "--fft-length", type="intx", default=512,
                          help="set the number of FFT bins [default=%default]")
        expert.add_option("","--strategy", type="int", default=1,
                          help="Set the spectrum access strategy (1: min block #, 2: min conflict, 0: trade-off between 1 and 2)")

    # Make a static method to call before instantiation
    add_options = staticmethod(add_options)

    def _print_verbage(self):
        """
        Prints information about the transmit path
        """
        print "Using TRAN d'board %s"    % (self.subdev.side_and_name(),)
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
#                           JELLO MAC
# /////////////////////////////////////////////////////////////////////////////

class ss_mac(object):

    def __init__(self, sender, start_time, verbose=False):
        self.verbose = verbose
        self.tb = None             # top block (access to PHY)
        self.sender = sender

        # linklab, init state
        self.state = "INIT"
        self.num_sent_req = 0
        
        self.demand = 0
        self.prev_demand = 0

        self.carrier_map = SYNC_MAP
        self.new_carrier_map = SYNC_MAP 
        self.prev_carrier_map = SYNC_MAP 
    
        self.pkts_since_reset = 0
        self.data_pkts = 0
        self.pktno = 0
        self.outage = 0

        self.start_time = start_time

    def read_traffic_trace(self, filename):
        FILE = open(filename) 
        trace = FILE.read()
        FILE.close()

        self.trace_demand = []
        self.trace_time = []
        self.trace_duration = []

        print "reading traffice trace", filename

        i = 0
        for line in trace.split('\n'):
            # print line
            fields = line.split('\t')
            if len(fields) >= 3:
                self.trace_time.append(float(fields[0]))
                self.trace_duration.append(float(fields[1]))
                self.trace_demand.append(int(fields[2]))
                # print self.trace_time[i], "\t", self.trace_duration[i], "\t", self.trace_demand[i]
                i += 1
        
        print "finished reading trace"

    def set_flow_graph(self, tb):
        self.tb = tb
        self.occupied_tones = self.tb.occupied_tones

        # linklab, freq
        self.freq = self.tb._tx_freq
        self.id = self.tb.id

        # linklab, read traffic trace
        if self.sender:
            tracefile = "%s/trace_%d" % (TRAFFIC_PATH, self.id)
            self.read_traffic_trace(tracefile)

        # spectrum strategy
        self.strategy = self.tb._strategy

        # print "carrier_map = ", self.carrier_map, "carrier_bin = ", self.carrier_bin, "demand = ", self.demand

    def phy_rx_callback(self, ok, payload, int_fo, frac_fo, time_sinr, freq_sinr): 
        """
        Invoked by thread associated with PHY to pass received packet up.

        @param ok: bool indicating whether payload CRC was OK
        @param payload: contents of the packet (string)
        """
        

        # if packet status is False, do nothing
        if not ok:
            return

        # if link_id is different, do nothing
        (link_id,)  = struct.unpack('!H', payload[2:4])
        if link_id != self.id:
            # print "self.id =", self.id, "link_id = ", link_id
            return

        # adjust freq offset after 10 successful rx
        freq_offset = int_fo+frac_fo/math.pi
        global n_right
        if ok:
            n_right += 1
        #if (not self.sender and ok and ((n_right == 1) or abs(freq_offset) > 0.5)):
        if (not self.sender and ok and (n_right == 1) ):
            self.tb._rx_freq = self.tb._rx_freq+freq_offset*64000000/self.tb._decim/self.tb._fft_length
            print "reset center freq:", self.tb._rx_freq
            self.reset_center_freq(self.tb._rx_freq) 
        
        
        # decode packet type
        (pkt_type,) = struct.unpack('!H', payload[0:2])

        # sender
        if self.sender:
            if pkt_type == ACK_PKT and self.state == "COOR":
                self.new_carrier_map = payload[4:4 + self.occupied_tones/4]
                
                print "LINK%d-CTL: ACK  \t%s\t%s\t%8.3f" % (self.id, self.carrier_map, self.new_carrier_map, time.time() - self.start_time)
                self.reset_carrier_map()
                self.req_timer.cancel()
                self.update_state("TRAN")

            return


        # receiver
        if not self.sender:
            # print "receiver got a pkt"
            self.last_pkt_time = time.time()
            if pkt_type == REQ_PKT and (self.state == "RECV" or self.state == "BOFF" or self.state == "SYNC"):
                
                # get sender sensing results
                self.avail_subc_str = payload[4:4 + self.occupied_tones/4]
                self.avail_subc_bin = subc_str2bin(self.avail_subc_str)
                
                # get request index
                (req_index,) = struct.unpack('!H', payload[4 + self.occupied_tones/4:6 + self.occupied_tones/4])
                
                # get demand from sender
                (self.demand,) = struct.unpack('!H', payload[6 + self.occupied_tones/4:8 + self.occupied_tones/4])
                print "LINK%d-CTL: REQ-%d\t%s\t%8.3f\t%s\t%d" % (self.id, req_index, self.carrier_map, time.time()-self.start_time, self.avail_subc_str, self.demand)
                self.update_state("COOR")
                
            elif pkt_type == DATA_PKT:
                (pktno,) = struct.unpack('!H', payload[4:6])
                
                self.pkts_since_reset += 1
                self.data_pkts += 1
                
                self.last_pkt_time = time.time()


                if pktno > 0:
                    loss_rate = 1 - float(self.data_pkts)/float(pktno)
                else:
                    loss_rate = 0
                print "LINK%d: DATA\t%6d\t%s\t%8.3fs" % (self.id, pktno, self.carrier_map, time.time()-self.start_time)


    def handle_backoff_timeout(self):
        # print "back off timer expires: received ", self.pkts_since_reset, "packets"
        if self.pkts_since_reset < 1:
            print "BOFF: Have not received enough pkts, should back off to ", self.prev_carrier_map
            self.new_carrier_map = self.prev_carrier_map
            self.reset_carrier_map()

    def start_backoff_timer(self):
        timeout = BOFF_TIMEOUT
        self.backoff_timer = Timer(timeout, self.handle_backoff_timeout)
        self.backoff_timer.start()


    def receiver_loop(self):
        print "This is a receiver of LINK%d" % (self.id)

        self.update_state("SYNC")

        self.pkts_since_reset = 0
        self.reset_time = time.time()
        self.last_pkt_time = time.time()
        self.traffic_index = 0

        while 1:
            if self.state == "RECV":
                while self.state == "RECV":
                    if time.time() - self.last_pkt_time > SYNC_TIMEOUT:
                        self.update_state("SYNC")
                    if time.time() - self.rx_start_time > BOFF_TIMEOUT and self.pkts_since_reset == 0:
                        self.update_state("BOFF")
                    pass

            elif self.state == "COOR":
                
                # receiver performs sensing
                used_bin_num = get_bin_num(subc_str2bin(self.carrier_map))
                
                # sense spectrum for two times
                if not self.tb.nosense:
                    avail_subc_bin = self.spectrum_sensing()
                else:
                    avail_subc_bin = numpy.ones(self.occupied_tones)
              
                # combine sensing results with sender 
                for i in range(0, len(avail_subc_bin)):
                    if avail_subc_bin[i] == 1 and self.avail_subc_bin[i] == 1:
                        self.avail_subc_bin[i] = 1
                    else:
                        self.avail_subc_bin[i] = 0

                self.avail_subc_str = subc_bin2str(self.avail_subc_bin)
                print "receier sensing results after combining: %s" % self.avail_subc_str


                new_carrier_map = find_new_carrier_map(self.carrier_map, avail_subc_bin, self.demand, self.strategy, 2, self.id)
                new_carrier_bin = subc_str2bin(new_carrier_map)
                allocated = get_bin_num(new_carrier_bin)
                print "allocated", allocated 
                print "self.demand", self.demand 
                print "self.prev_state", self.prev_state
                print "self.prev_demand", self.prev_demand
                print "self.outage", self.outage

                # linklab, record outage numbers
                if allocated < self.demand and (self.prev_state != "SYNC" or self.prev_demand == 0):
                    self.outage += 1
                    self.prev_demand = self.demand
                    print "LINK%d\tOUTAGE\t%d\t%d\t%d\t%d\t%8.3f" % (self.id, self.outage, self.traffic_index, allocated, self.demand, time.time())
                    print avail_subc_bin
                    print new_carrier_map, new_carrier_bin


                #if allocated >= self.demand:
                if allocated >= MIN_SUBC :
                    self.new_carrier_map = new_carrier_map
                    self.update_state("COOR")
                    self.send_ack()
                    self.reset_carrier_map()
                    self.update_state("RECV")
                    self.rx_start_time = time.time()
                    self.pkts_since_reset = 0

                elif self.prev_state == "SYNC":
                    self.update_state("SYNC")
                else:
                    print "Available spectrum cannot support traffic demand!"
                    #self.update_state("COOR")
                    self.send_ack()
                    self.reset_carrier_map()
                    self.update_state("RECV")
                    self.rx_start_time = time.time()
                    self.pkts_since_reset = 0


            elif self.state == "BOFF":

                print "BOFF: Have not received enough pkts, should back off to ", self.prev_carrier_map, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                self.new_carrier_map = self.prev_carrier_map
                self.reset_carrier_map()
                print "backoff start time is %8.3f" % (time.time())
                self.backoff_time = time.time()

                while self.state == "BOFF":
                    if time.time() - self.backoff_time > SYNC_TIMEOUT:
                        self.update_state("SYNC")
                    pass

            elif self.state == "SYNC":
                if self.carrier_map != SYNC_MAP: 
                    self.new_carrier_map = SYNC_MAP 
                    self.reset_carrier_map()
                    print "reset carrier_map in sync is finished"
                    
                while self.state == "SYNC":
                    pass

        # linklab, shall not return
        return

    def sender_loop(self):
        print "This is a sender of LINK%d" % (self.id)
        self.update_state("SYNC")

        self.outage = 0
        self.traffic_index = 0
        self.start_traffic_timer()
        self.demand = self.generate_traffic() 
        self.prev_demand = self.demand

        while 1:
            if self.state == "TRAN":
                self.pktno += 1
                self.send_data(self.pktno)
                
            elif self.state == "SENS":
                used_bin_num = get_bin_num(subc_str2bin(self.carrier_map))
                
                # sense spectrum for two times
                if not self.tb.nosense:
                    self.avail_subc_bin = self.double_spectrum_sensing()
                else:
                    self.avail_subc_bin = numpy.ones(self.occupied_tones)
               
                self.avail_subc_str = subc_bin2str(self.avail_subc_bin)
                
                # enter COOR state and send demand size and sensing results to receiver
                self.update_state("COOR")

            elif self.state == "COOR":
                # reset the sent request counter
                self.num_sent_req = 1
                
                # send request with sensing results and demand size
                self.send_request()
                self.wait_until_txq_empty()

                # record the request sent time
                self.req_sent_time = time.time()
                self.start_req_timer()
                while self.state == "COOR":
                    pass 
            
            elif self.state == "SYNC":
                if self.carrier_map != SYNC_MAP: 
                    self.new_carrier_map = SYNC_MAP 
                    self.reset_carrier_map()
                self.update_state("SENS")

            else:
                pass
        
        # linklab, shall not return
        return

    # Main loop for MAC
    def main_loop(self):

        self.carrier_map = SYNC_MAP 
        self.new_carrier_map = SYNC_MAP
        self.prev_carrier_map = SYNC_MAP

        if not self.sender:
            self.receiver_loop()
        else:
            self.sender_loop()
        return
        
    def wait_until_txq_empty(self): 
        while not self.tb.txpath.ofdm_tx._pkt_input.msgq().empty_p():
            pass

    def generate_traffic(self):
        demand = self.trace_demand[self.traffic_index]
        print "new traffic demand is", demand, "index is", self.traffic_index
        self.traffic_index += 1
        return demand


    def start_traffic_timer(self):
        timeout = self.trace_duration[self.traffic_index]
        print "new traffic duration is", timeout, "index is", self.traffic_index
        self.traffic_timer = Timer(timeout, self.handle_traffic_timeout)
        self.traffic_timer.start()

    def handle_traffic_timeout(self):
        # restart traffic timer
        self.start_traffic_timer()
        
        self.prev_demand = self.demand
        self.demand = self.generate_traffic()
        if self.state == "TRAN":
            used_bin_num = get_bin_num(subc_str2bin(self.carrier_map))
            print "\nTraffic demand changes from %d to %d" % (used_bin_num, self.demand)
            if self.demand == 0:
                self.update_state("SYNC")
            else:
                self.update_state("SENS")


    # linklab, reset receiver
    def reset_receiver(self):
        adc_rate = self.tb.u_src.adc_rate()

        self.tb.u_src.set_decim_rate(self.tb._decim)

        # determine the daughterboard subdevice we're using
        if self.tb._rx_subdev_spec is None:
            self.tb._rx_subdev_spec = usrp.pick_rx_subdevice(self.tb.u_src)
        self.tb.subdev = usrp.selected_subdev(self.tb.u_src, self.tb._rx_subdev_spec)
        self.tb.subdev_src = self.tb.subdev

        self.tb.u_src.set_mux(usrp.determine_rx_mux_value(self.tb.u_src, self.tb._rx_subdev_spec))


    # linklab
    def reset_center_freq(self, freq):
        print "LINK", self.id, "resets central freq to ", n2s(freq), "Hz"
        self.tb.set_snk_freq(freq)
        self.tb.set_src_freq(freq)


    def reset_carrier_map(self):
        # wait until tx queue is empty
        self.wait_until_txq_empty()
        
        # update carrier map for both tx and rx
        print "resetting rx carrier map to", self.new_carrier_map
        self.tb.rxpath.reset_carrier_map(self.new_carrier_map)

        print "resetting tx carrier map to", self.new_carrier_map
        self.tb.txpath.reset_carrier_map(self.new_carrier_map)
        
        # update variable for map and bins
        self.prev_carrier_map = self.carrier_map
        self.carrier_map = self.new_carrier_map
        #self.carrier_bin = subc_str2bin(self.new_carrier_map)


    def update_state(self, state):
        print "LINK%d: %s =====> %s\t%s\t%8.3f" % (self.id, self.state, state, self.carrier_map, time.time() - self.start_time)
        # update variable
        self.prev_state = self.state
        self.state = state
        

    def handle_req_timeout(self):
        if self.state == "COOR":
            #print "The ACK of the %d REQ is not received in %2.1f sec, retrans REQ" % (self.num_sent_req, REQUEST_TIMEOUT)
            if self.num_sent_req < MAX_REQ_RETRY:
                self.num_sent_req += 1
                self.send_request()
                self.start_req_timer()
            else:
                self.update_state("SYNC")
        else:
            #print "Got ACK"
            pass

    def start_req_timer(self):
        timeout = REQUEST_TIMEOUT
        self.req_timer = Timer(timeout, self.handle_req_timeout)
        self.req_timer.start()


    def double_spectrum_sensing(self):
        # linklab, sense two time when traffic increases
        avail_subc_bin = self.spectrum_sensing()
        #if self.demand > used_bin_num:
        #print "demand > current allocated, sense two times"
        if self.carrier_map != SYNC_MAP: 
            for i in range(0, random.randint(10, 20)):
                self.pktno += 1
                self.send_data(self.pktno)
        else:
            # time.sleep(random.uniform(0.1,2))
            random_time = random.uniform(0.1,2)
            t1 = time.time()
            while time.time() < t1 + random_time:
                pass

        
        avail_subc_bin2 = self.spectrum_sensing()
        for i in range(0, len(avail_subc_bin)):
            if avail_subc_bin2[i] == 0:
                avail_subc_bin[i] = 0
        return avail_subc_bin


    # linklab, spectrum sensing
    def spectrum_sensing(self):
        # print "LINK", self.id, "starts sensing"
        # linklab, do not sense until the tx queue is empty
        start_time = time.time()
        self.wait_until_txq_empty()
    
        avail_subc_bin = self.tb.senspath.get_avail_carriers()
        avail_subc_str = subc_bin2str(avail_subc_bin)


        print "LINK%d: Sensing Results\t%s\t%8.3f\t%8.3f" %  (self.id, avail_subc_str, time.time()-start_time, time.time()-self.start_time)
        #print "LINK%d: %s =====> %s\t%s\t%8.3f" % (self.id, self.state, state, self.carrier_map, time.time() - self.start_time)
        #print_carrier_bin(avail_subc_bin)

        return avail_subc_bin

    
    # linklab, sender sends data packets
    def send_data(self, pktid):
        sys.stderr.write('.')
        payload = struct.pack('!H', pktid) + (DATA_PKT_SIZE - 2) * chr(pktid & 0x00)
        self.send_pkt(DATA_PKT, payload)
        # time.sleep(0.1)

    # linklab, sender sends request to receiver
    def send_request(self):
        print "LINK%d-CTL: REQ-%d\t%s\t%8.3f\t%s" % (self.id, self.num_sent_req, self.avail_subc_str, time.time() - self.start_time, self.new_carrier_map)
        payload = self.avail_subc_str + struct.pack('!H', self.num_sent_req) + struct.pack('!H', self.demand)+ (REQ_PKT_SIZE - 64) *chr(0x11)
        
        # linklab send req for 3 times
        self.send_pkt(REQ_PKT, payload)
        self.send_pkt(REQ_PKT, payload)
        self.send_pkt(REQ_PKT, payload)
        
    # linklab, receiver sends ACK to sender
    def send_ack(self):
        print "LINK%d-CTL: ACK  \t%s\t%s\t%8.3f" % (self.id, self.carrier_map, self.new_carrier_map, time.time() - self.start_time)
        #payload = (ACK_PKT_SIZE - 4) * chr(ACK_PKT & 0x00)
        payload = self.new_carrier_map + (ACK_PKT_SIZE - 64) *chr(0x22)
        
        # linklab send ack for 3 times
        self.send_pkt(ACK_PKT, payload) 
        self.send_pkt(ACK_PKT, payload) 
        self.send_pkt(ACK_PKT, payload) 
        
        # not return until tx queue is empty
        self.wait_until_txq_empty()

    # linklab, generic pkt send
    def send_pkt(self, pkt_type, payload):
        packet = struct.pack('!H', pkt_type) + struct.pack('!H', self.id) + payload
        self.tb.txpath.send_pkt(packet)


# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////


def main():

    global n_right
    n_right = 0 

    start_time = time.time()

    parser = OptionParser (option_class=eng_option, conflict_handler="resolve")
    expert_grp = parser.add_option_group("Expert")

    parser.add_option("-m", "--modulation", type="choice", choices=['bpsk', 'qpsk'],
                      default='bpsk',
                      help="Select modulation from: bpsk, qpsk [default=%%default]")
    expert_grp.add_option("-c", "--carrier-threshold", type="eng_float", default=40,
                      help="set carrier detect threshold (dB) [default=%default]")

    parser.add_option("-v","--verbose", action="store_true", default=False)

    # linklab, add option to indicate sender or receiver
    parser.add_option("-s","--sender", action="store_true", default=False)
    parser.add_option("-r","--receiver", action="store_true", default=False)


    usrp_graph.add_options(parser, expert_grp)
    transmit_path.add_options(parser, expert_grp)
    receive_path.add_options(parser, expert_grp)
    blks2.ofdm_mod.add_options(parser, expert_grp)
    blks2.ofdm_demod.add_options(parser, expert_grp)

    fusb_options.add_options(expert_grp)

    (options, args) = parser.parse_args ()
    options.carrier_map = SYNC_MAP
    
    if len(args) != 0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if options.rx_freq is None or options.tx_freq is None:
        sys.stderr.write("You must specify -f FREQ or --freq FREQ\n")
        parser.print_help(sys.stderr)
        sys.exit(1)
    
	# linklab, check if the current node is either a sender or a receiver
    if options.sender and options.receiver:
        sys.stderr.write("You cannot specify both sender and receiver\n")
        sys.exit(1)
    if (not options.sender) and (not options.receiver):
        sys.stderr.write("You must specify either sender or receiver\n")
        sys.exit(1)
    #

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
    
    #print "fusb_block_size =", options.fusb_block_size
    #print "fusb_nblocks    =", options.fusb_nblocks

    # instantiate the MAC
    # linklab, use ssma instead of csma
	mac = ss_mac(options.sender, start_time, verbose=True)

    # update freq_offset
    options.rx_freq += get_freq_offset()
    options.tx_freq += get_freq_offset()
    print "RX frequency", options.rx_freq
    print "TX frequency", options.tx_freq

    # build the graph (PHY)
    tb = usrp_graph(mac.phy_rx_callback, options)

    mac.set_flow_graph(tb)    # give the MAC a handle for the PHY

    #if fg.txpath.bitrate() != fg.rxpath.bitrate():
    #    print "WARNING: Transmit bitrate = %sb/sec, Receive bitrate = %sb/sec" % (
    #        eng_notation.num_to_str(fg.txpath.bitrate()),
    #        eng_notation.num_to_str(fg.rxpath.bitrate()))
             
    print "modulation:     %s"   % (options.modulation,)
    print "freq:           %s"      % (eng_notation.num_to_str(options.tx_freq))
    # print "bitrate:        %sb/sec" % (eng_notation.num_to_str(tb.txpath.bitrate()),)
    # print "samples/symbol: %3d" % (tb.txpath.samples_per_symbol(),)
    # print "interp:         %3d" % (tb.txpath.interp(),)
    # print "decim:          %3d" % (tb.rxpath.decim(),)


    tb.start()    # Start executing the flow graph (runs in separate threads)

    mac.main_loop()    # don't expect this to return...
    while 1:
        pass
               

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print "KeyboardInterrupt in main"
        sys.exit(1)
        pass
    finally:
        sys.exit(1)
