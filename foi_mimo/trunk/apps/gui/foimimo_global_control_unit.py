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

from foimimo_tb_options import *
from foimimo_top_blocks import *
from gui.foimimo_sendThread import Send_thread

class global_ctrl_unit:
    def __init__(self,mode):
        # Init values of global variables:
        self.mode = mode #rx_gui, tx_gui, 
        self.running = False
        self.default_options()
        self.focus_view = ""
        self.GUIAlive = True      
        self.reset_statistics()
    
    def reset_statistics(self):
        self.npkt_rcvd = 0
        self.npkt_right = 0
        self.nbit_rcvd = 0
        self.nbit_right = 0
        self.nr_bad_headers = 0
    
    def get_pkt_statistics(self):
        return (self.npkt_rcvd,self.npkt_right,self.nr_bad_headers)
    
    def get_bit_statistics(self):
        return (self.nbit_rcvd,self.nbit_right)
    
    def update_statistics(self,npkt_rcvd,npkt_right,nbit_rcvd,nbit_right):
        self.npkt_rcvd += npkt_rcvd
        self.npkt_right += npkt_right
        self.nbit_rcvd += nbit_rcvd
        self.nbit_right += nbit_right

    def update_bad_header(self, new_bad_headers):
        self.nr_bad_headers += new_bad_headers
        
    def start_flowgraph(self,rx_callback='',bad_header_callback='',all_sinks=0,send_data=[]):
        self.reset_statistics()      
        self.rx_callback = rx_callback
        self.bad_header_callback = bad_header_callback
        self.all_sinks = all_sinks
        # Create a top block and a sender thread with given options:
        if self.mode == "rx_gui":
            self.top_block = 0
            self.top_block = top_block_rx_gui(self.rx_callback,self.bad_header_callback,self.options,self.all_sinks)
        elif self.mode == "tx_gui":
            self.top_block = 0
            self.top_block = top_block_tx_gui(self.options)
            self.sender = 0
            print "Length of send data", len(send_data)
            self.sender = Send_thread(self, self.top_block.send_pkt, send_data)    # Start sender thread
            self.sender.start()
        else:
            print "unknown mode"
        
        # start the top block and the sender thread:
        self.top_block.start()
        self.running = True
    
    def stop_flowgraph(self):
        if self.running:
            self.top_block.stop()
            self.top_block.wait()
            self.top_block = 0
            self.running = False
        else:
            pass
    
    def get_running(self):
        # Return values for self.running
        return self.running
    def set_running(self,value):
        # Set values for self.running
        self.running = value
        
    def get_options_mimo(self):
        # Return values for self.options.siso
        return not self.options.siso
    def set_options_mimo(self,value):
        # Get values for self.options.siso
        if value: # bool or int value
            self.options.siso = 0
        else:
            self.options.siso = 1
        
    def get_options(self):
        # Return values for self.options
        return self.options
    def set_options(self,options_given):
        # Set values for self.options
        self.options = options_given
    
    def default_options(self):
        # Set and return default values for self.options
        if self.mode == "rx_gui":
            self.options = tb_options_rx_gui()
        elif self.mode == "tx_gui":
            self.options = tb_options_tx_gui()
        else:
            print "unknown mode"
        return self.options

    def set_focus_view(self,view):
        self.focus_view = view
    def get_focus_view(self):
        return self.focus_view
    def exit(self):
        self.GUIAlive = False
