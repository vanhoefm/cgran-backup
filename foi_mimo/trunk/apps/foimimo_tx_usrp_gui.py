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

from gnuradio import gr
from gnuradio import uhd
import time, struct, sys, math
import gnuradio.gr.gr_threading as _threading
import wx

# import local gui classes
from gui.foimimo_gui_main_frame_tx import GUI_thread, main_frame
from gui.foimimo_events import *
from gui.foimimo_global_control_unit import global_ctrl_unit

VERBOSE = 0

# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():
    global n_rcvd, n_right
    
    ###--- Realtime scheduling ---###
    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
            print "Warning: failed to enable realtime scheduling"

    ###--- GUI ---###
    app = wx.App(False)
    global_ctrl = global_ctrl_unit("tx_gui")
    frame = main_frame(global_ctrl,app)
    gui = GUI_thread(app)
    
    while global_ctrl.GUIAlive:    
        time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

