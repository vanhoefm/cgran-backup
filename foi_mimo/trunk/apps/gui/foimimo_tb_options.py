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

class tb_options_rx_gui:
    def __init__(self):
        #Normal:
        self.size = float(401)
        self.npixels = float(1000000.0)
        self.input_filename = str("")
        self.output_filename_mimo = str("")
        self.output_filename_siso = str("")
        self.snr = float(30) #Not in GUI, used by ofdm receiver, estimated snr for cp synch
        self.discontinuous = int(0)
        self.tx_amplitude = float(0.2)
        self.modulation = str("qpsk")
        self.code_rate = str("3/4")    # will not be set in gui, only first choice of dropdown works, need to change order in foimimo_ctrl_panel.py
        self.siso = 1
        self.verbose = 0
        self.write_all = 1
        self.per_avg_nr = int(20)        
        self.image_mode = 1     # only one of the three modes can be set to 1
        self.file_mode = 0
        self.benchmark_mode = 0
        #Expert:
        self.fft_length = int(512)
        self.occupied_tones = int(200)
        self.cp_length = int(128)
        self.log = 0
        #USRP:
        self.usrp_rx = True
        self.usrp_tx = False
        self.usrp_addr0 = str("192.168.20.2")
        self.usrp_addr1 = str("192.168.30.2")
        self.center_freq = float(433000000.0)
        self.sample_rate = float(3125000.0)
        self.gain_rx = float(15.0)
        self.gain_tx = float(25.0)
        
class tb_options_tx_gui:
    def __init__(self):
        #Normal:
        self.size = float(401)
        self.npixels = float(1000000.0)
        self.input_filename = str("")
        self.output_filename_mimo = str("")
        self.output_filename_siso = str("")
        self.snr = float(30) #Not in GUI, used by ofdm receiver, estimated snr for cp synch
        self.discontinuous = int(0)
        self.tx_amplitude = float(0.2)
        self.modulation = str("qpsk")
        self.code_rate = str("3/4")    # will not be set in gui, only first choice of dropdown works, need to change order in foimimo_ctrl_panel.py
        self.siso = 1
        self.verbose = 0
        self.write_all = 1
        self.per_avg_nr = int(20)
        self.image_mode = 1     # only one of the three modes can be set to 1
        self.file_mode = 0
        self.benchmark_mode = 0
        #Expert:
        self.fft_length = int(512)
        self.occupied_tones = int(200)
        self.cp_length = int(128)
        self.log = 0
        #USRP:
        self.usrp_rx = False
        self.usrp_tx = True
        self.usrp_addr0 = str("192.168.30.2")
        self.usrp_addr1 = str("192.168.40.2")
        self.center_freq = float(433000000.0)
        self.sample_rate = float(3125000.0)
        self.gain_rx = float(15.0)
        self.gain_tx = float(25.0)
