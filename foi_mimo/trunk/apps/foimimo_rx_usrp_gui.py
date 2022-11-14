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

from foimimo.foi_mimo_rx_path import mimo_receive_path as mimo_rx_path
from foimimo.foi_siso_rx_path import siso_receive_path as siso_rx_path

from gnuradio import gr
from gnuradio import uhd
import time, struct, sys, math
import wx 

# import local gui classes
from gui.foimimo_gui_main_frame_rx import GUI_thread, main_frame
from gui.foimimo_events import *
from gui.foimimo_global_control_unit import global_ctrl_unit

VERBOSE = 0
                
# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main(): 
    global per_history  # packet error history
    per_history = []    # init
       
    def rx_callback(ok, payload):
        global per_history                  # packet error history
        options = global_ctrl.get_options()
        
        # Received one packet
        bit_rcvd = 0
        bit_right = 0
        n_rcvd = 1 
        n_right = 0
        if ok:
            n_right = 1
        
        # Calculate a floating packet error rate
        if ok: per_history.append(1)
        else: per_history.append(0)
        per_avg_nr = options.per_avg_nr # nr of packets to make average over
        if n_rcvd > per_avg_nr: per_history.pop(0)
        per_avg = (len(per_history)-sum(per_history))      # none=0, max=per_avg_nr(eller n_rcvd)
        per_avg = 100*(float(per_avg)/len(per_history))    # in %
        
        # Post event to update the per graph in the gui:
        event = per_event(myEVT_PER,-1)
        event.set_per_val(per_avg)
        wx.PostEvent(frame.tech_panel.on_per_event(event),event)
        
        # Post event to update received data image in the gui:
        if (ok or options.write_all):
            event = rcvd_data_event(myEVT_PKT,-1)
            event.set_data(payload)
            wx.PostEvent(frame.demo_panel.on_rcvd_data_event(event),event)
                        
        # Receive data and print to screen
        if options.image_mode:  
                 
            if VERBOSE:
                printlst = list()
                for x in payload:
                        t = hex(ord(x)).replace('0x', '')
                        if(len(t) == 1):
                                t = '0' + t
                        printlst.append(t)
                printable = ''.join(printlst)
        
                print printable
                print "\n"
                sys.stdout.flush()
                    
        # Receive data and write to file
        elif options.file_mode:
            if VERBOSE:
                print "TB: Got data with len", len(payload)," status:",ok
                # Write payload to file:
            if options.mimo:
                out_file = open(options.output_filename_mimo,mode="ab")
            else:
                out_file = open(options.output_filename_siso,mode="ab")
            if (ok or options.write_all):
                out_file.write(payload) 
            out_file.close()
            #print "n_rcvd: %d \t n_right: %d" % (n_rcvd, n_right)
        
        # Receive data and print to screen        
        elif options.benchmark_mode:          
            if VERBOSE:
                printlst = list()
                for x in payload:
                        t = hex(ord(x)).replace('0x', '')
                        if(len(t) == 1):
                                t = '0' + t
                        printlst.append(t)
                printable = ''.join(printlst)
        
                print printable
                print "\n"
                sys.stdout.flush()
        elif options.ber_mode:
            bit_rcvd += 8*len(payload[2:])
            if ok:
                (pktno,) = struct.unpack('!H', payload[0:2])
                bit_right += 8*len(payload[2:])
            else:
                for x in payload[2:]:
                    xored = ord(x)^0 # we send 0 (zeros as payload)
                    if xored == 0:
                        bit_right += 8
                    else:
                        #convert to a string of '0' and '1'
                        xored = bin(xored) #obs, count ones not zeros
                        xored = xored[2:]
                        bit_right = 8
                        for b in xored:
                            bit_right -=int(b)
        global_ctrl.update_statistics(n_rcvd, n_right, bit_rcvd, bit_right) 

    ###--- Bad header counter - Callback ---###
    def bad_header_callback():
        global_ctrl.update_bad_header(1) 
                               
    ###--- Reset all counters etc. ---###
    def reset_main_variables():
        global n_rcvd, n_right
        n_rcvd = 0
        n_right = 0
        global per_history  # packet error history
        per_history = []    # init
        
    ###--- Realtime scheduling ---###
    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
            print "Warning: failed to enable realtime scheduling"

    ###--- GUI ---###
    app = wx.App(False)
    global_ctrl = global_ctrl_unit("rx_gui") 
    frame = main_frame(rx_callback,bad_header_callback,reset_main_variables,global_ctrl,app)
    gui = GUI_thread(app)
    
    while global_ctrl.GUIAlive:
        pass
        time.sleep(1)

    # Never reached...

###--- Start program ---###
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

