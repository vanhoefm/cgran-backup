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

from gnuradio import gr,uhd
from foimimo import *

import math

from foimimo.foi_mimo_rx_path import mimo_receive_path as mimo_rx_path
from foimimo.foi_mimo_tx_path import mimo_transmit_path as mimo_tx_path
from foimimo.foi_siso_rx_path import siso_receive_path as siso_rx_path
from foimimo.foi_siso_tx_path import siso_transmit_path as siso_tx_path

class top_block_rx_gui(gr.top_block):

    def __init__(self, rx_callback, bad_header_callback, options, all_gui_sinks):
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
            device_addr = "addr="+options.usrp_addr0
            print "Using USRP unit with device adress: ",device_addr
            self.uhd_usrp_source = uhd.single_usrp_source(
            device_addr=device_addr,
            io_type=uhd.io_type.COMPLEX_FLOAT32,
            num_channels=1,
            )
            self.uhd_usrp_source.set_clock_config(uhd.clock_config.external());
            self.uhd_usrp_source.set_time_next_pps(uhd.time_spec())
            self.uhd_usrp_source.set_samp_rate(options.sample_rate)
            self.uhd_usrp_source.set_center_freq(options.center_freq, 0)
            self.uhd_usrp_source.set_gain(options.gain_rx, 0)

            self.foi_rxpath = siso_rx_path(rx_callback, bad_header_callback, options)

        else:
            device_addr = "addr0="+options.usrp_addr0+", addr1="+options.usrp_addr1
            print "Using USRP units with device adress: ",device_addr
            self.uhd_usrp_source = uhd.multi_usrp_source(
                device_addr=device_addr,
                io_type=uhd.io_type.COMPLEX_FLOAT32,
                num_channels=2,
                )
            self.antenna0 = self.uhd_usrp_source.get_antenna(0)
            self.antenna1 = self.uhd_usrp_source.get_antenna(1)
            print "Using antenna " , self.antenna0, " and ", self.antenna1
            # Set usrp source parameters
            self.uhd_usrp_source.set_clock_config(uhd.clock_config.external(), uhd.ALL_MBOARDS);
            self.uhd_usrp_source.set_time_unknown_pps(uhd.time_spec())
            self.uhd_usrp_source.set_samp_rate(options.sample_rate)
            self.uhd_usrp_source.set_center_freq(options.center_freq, 0)
            self.uhd_usrp_source.set_center_freq(options.center_freq, 1)
            self.uhd_usrp_source.set_gain(options.gain_rx, 0)
            self.uhd_usrp_source.set_gain(options.gain_rx, 1)
    
            self.foi_rxpath = mimo_rx_path(rx_callback, bad_header_callback, options)

        # GUI sinks:
        # Get sinks in list of tuples all_gui_sinks
        # Tuples consist of (sink,name)
        i = [x[1] for x in all_gui_sinks].index("fft_channel_filter_0")
        self.wxgui_fftsink_chanfilt0 = all_gui_sinks[i][0]
        i = [x[1] for x in all_gui_sinks].index("fft_channel_filter_1")
        self.wxgui_fftsink_chanfilt1 = all_gui_sinks[i][0]
        i = [x[1] for x in all_gui_sinks].index("constellation_fft_0")
        self.wxgui_constellationsink_fft0 = all_gui_sinks[i][0]
        i = [x[1] for x in all_gui_sinks].index("constellation_fft_1")
        self.wxgui_constellationsink_fft1 = all_gui_sinks[i][0]
        i = [x[1] for x in all_gui_sinks].index("constellation_frame_acq")
        self.wxgui_constellationsink_frameacq = all_gui_sinks[i][0]
        i = [x[1] for x in all_gui_sinks].index("constellation_frame_sink")
        self.wxgui_constellationsink_framesink = all_gui_sinks[i][0]
            
        ##################################################
        # Connections
        ##################################################
        if options.siso:
            self.connect((self.uhd_usrp_source, 0), (self.foi_rxpath, 0))
            
            # Connect FFT sink after the channel filter
            self.connect((self.foi_rxpath.ofdm_siso_demod.ofdm_recv.chan_filt, 0), (self.wxgui_fftsink_chanfilt0, 0))
            # Connect constellation plots after channel FFT, after frame acq. and after end
            self.connect((self.foi_rxpath.ofdm_siso_demod.ofdm_recv.fft_demod, 0), (self.wxgui_constellationsink_fft0, 0))
            self.connect((self.foi_rxpath.ofdm_siso_demod.ofdm_recv.ofdm_frame_acq, 0), (self.wxgui_constellationsink_frameacq, 0))       
            if options.code_rate == "":
                self.connect((self.foi_rxpath.ofdm_siso_demod.ofdm_demod, 0), (self.wxgui_constellationsink_framesink, 0))
            else:
                self.connect((self.foi_rxpath.ofdm_siso_demod.ofdm_demod, 0),  gr.stream_to_vector(gr.sizeof_gr_complex,options.occupied_tones), (self.wxgui_constellationsink_framesink, 0))
                  
        else:
            self.connect((self.uhd_usrp_source, 0), (self.foi_rxpath, 0))
            self.connect((self.uhd_usrp_source, 1), (self.foi_rxpath, 1))

            # Connect FFT sinks after the channel filters in both channels
            self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_recv.chan_filt[0], 0), (self.wxgui_fftsink_chanfilt0, 0))
            self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_recv.chan_filt[1], 0), (self.wxgui_fftsink_chanfilt1, 0))
            # Connect constellation plots after both channel FFTs, after frame acq. and after end => 4 plots
            self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_recv.fft_demod[0], 0), (self.wxgui_constellationsink_fft0, 0))
            self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_recv.fft_demod[1], 0), (self.wxgui_constellationsink_fft1, 0))
            self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_recv.ofdm_frame_acq, 1), (self.wxgui_constellationsink_frameacq, 0))        
            if options.code_rate == "":
                self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_demod, 0), (self.wxgui_constellationsink_framesink, 0))
            else:
                self.connect((self.foi_rxpath.ofdm_mimo_demod.ofdm_demod, 0), gr.stream_to_vector(gr.sizeof_gr_complex,options.occupied_tones),(self.wxgui_constellationsink_framesink, 0))

class top_block_tx_gui(gr.top_block):

    def __init__(self, options):
        gr.top_block.__init__(self)
        
        ##################################################
        # Variables
        ##################################################
        
        code = {"":[1,1],
                "3/4":[3.0, 4.0],
                "2/3":[2.0, 3.0],
                "1/3":[1.0, 3.0]}
        (k,n) = code[options.code_rate]
                       
        symbols_per_packet = int(math.ceil(((((options.size-4)*(n/k))+4) * 8.0) / (2.0*options.occupied_tones)))
        samples_per_packet = (symbols_per_packet+2) * (options.fft_length+options.cp_length)
        print "\nSymbols per Packet: ", symbols_per_packet
        print "Samples per Packet: " + str(samples_per_packet) + "\n"
        
        if options.discontinuous:
            stream_size = [100000, int(options.discontinuous*samples_per_packet)]
        else:
            stream_size = [0, 100000]
        

        ##################################################
        # Blocks
        ##################################################
        if options.siso:
            print "SISO"
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
            self.uhd_usrp_sink.set_center_freq(options.center_freq, 0)
            self.uhd_usrp_sink.set_gain(options.gain_tx, 0)
            antenna_0 = self.uhd_usrp_sink.get_antenna(0)
            print "Transmitting on antenna0:",antenna_0

            self.foi_txpath = siso_tx_path(options)

        else:
            print "MIMO"
            device_addr = "addr0="+options.usrp_addr0+", addr1="+options.usrp_addr1
            print "Using USRP units with device adress: ",device_addr
            self.uhd_usrp_sink = uhd.multi_usrp_sink(
                device_addr=device_addr,
                io_type=uhd.io_type.COMPLEX_FLOAT32,
                num_channels=2,
                )
            # Set usrp sink parameters
            self.uhd_usrp_sink.set_clock_config(uhd.clock_config.external(), uhd.ALL_MBOARDS);
            self.uhd_usrp_sink.set_time_unknown_pps(uhd.time_spec())
            self.uhd_usrp_sink.set_samp_rate(options.sample_rate)
            self.uhd_usrp_sink.set_center_freq(options.center_freq, 0)
            self.uhd_usrp_sink.set_center_freq(options.center_freq, 1)
            self.uhd_usrp_sink.set_gain(options.gain_tx, 0)
            self.uhd_usrp_sink.set_gain(options.gain_tx, 1)
            antenna_0 = self.uhd_usrp_sink.get_antenna(0)
            antenna_1 = self.uhd_usrp_sink.get_antenna(1)
            print "Transmitting on antenna0:",antenna_0,"and antenna1:",antenna_1

            self.foi_txpath = mimo_tx_path(options)


        ##################################################
        # Connections
        ##################################################
        if options.siso:
            self.connect((self.foi_txpath, 0), (self.uhd_usrp_sink, 0))
        else:
            self.connect((self.foi_txpath, 0), (self.uhd_usrp_sink, 0))
            self.connect((self.foi_txpath, 1), (self.uhd_usrp_sink, 1))

    ###--- Transmitter - Send packet ---###
    def send_pkt(self,payload='', eof=False):
        return self.foi_txpath.send_pkt(payload, eof)
