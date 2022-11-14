#!/usr/bin/env python
#
# Copyright 2005,2006,2007 Free Software Foundation, Inc.
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

from gnuradio import gr, gru, modulation_utils,packet_utils
from gnuradio import usrp
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from optparse import OptionParser
#from gnuradio import dectv1

import random
import struct
import sys

# from current dir
from receive_path_dect import receive_path
import fusb_options
from dect_mac_parser import extract_Q_system_information
from packet_utils_dect import conv_1_0_string_to_binary_string


#import os
#print os.getpid()
#raw_input('Attach and press enter: ')

global n_rcvd, n_right,nchan
global tb,file_write,print_q,print_n,print_m,print_c,print_p



class my_top_block(gr.top_block):
    def __init__(self, demodulator, rx_callback,access_code, options):
        gr.top_block.__init__(self)
        self.rxpath = receive_path(demodulator, rx_callback,access_code, options) 
        self.connect(self.rxpath)


# /////////////////////////////////////////////////////////////////////////////
#                                   main
# /////////////////////////////////////////////////////////////////////////////

def main():
    global n_rcvd, n_right,nchan,rfp_pp
    global tb,file_write,print_q,print_n,print_m,print_c,print_p

    n_rcvd = 0
    n_right = 0


    
    def rx_callback(ok, payload):
         global n_rcvd, n_right,nchan,rfp_pp
         global file_write,print_q,print_n,print_m,print_c,print_p,FILE,FILE1
 
         str1=packet_utils.conv_packed_binary_string_to_1_0_string(payload)
         fer =1e0
         n_rcvd+=1
         #print "callback "
         if ok:
           n_right += 1
         if  ok:
	   if file_write:
	     #puppa=packet_utils.conv_1_0_string_to_packed_binary_string(str1[64:384])
	     FILE.write(conv_1_0_string_to_binary_string(str1[64:384]))
	     #FILE.write(str1[64:384].decode("hex"))
	     FILE1.write(str1[64:384])
#/////////////////////////////////////////////////////////////////////////////
#                     N TYPE  IDENTITY           
	   if str1[:3] == "011" and print_n:
	     fer=float(n_right)/float(n_rcvd)
	     print "N type identity head: " , str1[:8],"Ch. ",nchan
	     if rfp_pp:
               print "RFP identifier=%x" % int(str1[8:48],2),"fer: " , fer
	     else:
               print "PP identifier=%x" % int(str1[8:48],2),"fer: " , fer
             #print "crc=%x" % int(str1[48:64],2)
	     print
#/////////////////////////////////////////////////////////////////////////////
#                     Q TYPE  BROADCAST           
	   if str1[:3] == "100" and print_q:
             print "Q type base station broadcast"
	     print "channel =",nchan
	     print "header=" , str1[:8]
             print "control bits=%x" % int(str1[8:48],2)
             #print "crc=", str1[48:64]
	     extract_Q_system_information(str1[8:48])
	     print
#///////////////////////////////////////////////////////////////////////////////////	     
#		  C type traffic channel
	   if str1[:3] == "000" and print_c:
             print "C type traffic channel 000  Ch. ",nchan
	     print "A field:",str1[:64]
	     if str1[4:7] == "111":
	       print "B field:",str1[64:384]
	       print "X field (CRC):",str1[384:388]
	       print
	   if str1[:3] == "001" and print_c:
             print "C type traffic channel 001  Ch. ",nchan
	     print "A field:",str1[:64]
	     if str1[4:7] != "111":
	       print "B field:",str1[64:384]
	       print "X field (CRC):",str1[384:388]
	       print
#//////////////////////////////////////////////////////////////////////////////////
	   if str1[:3] == "111" and print_p:
             print "P type base station paging message"
             print "channel =",nchan
	     print "header=" , str1[:8]
             print "control bits=%x" % int(str1[8:48],2)
	     fer=float(n_right)/float(n_rcvd)
	     print "fer: " , fer
#/////////////////////////////////////////////////////////////////////////////////////
	   if str1[:3] == "110" and print_m:
             print "M MAC control message"
	   #if str1[4:7] == "111":
             #print "no B field"
           

    demods = modulation_utils.type_1_demods()

    # Create Options Parser:
    parser = OptionParser (option_class=eng_option, conflict_handler="resolve")
    expert_grp = parser.add_option_group("Expert")

    parser.add_option("-m", "--modulation", type="choice", choices=demods.keys(), 
                      default='gmsk',
                      help="Select modulation from: %s [default=%%default]"
                            % (', '.join(demods.keys()),))
    parser.add_option("-w", "--which", type=int, default=1,
                      help="select which USRP (0, 1, ...) default is %default",metavar="NUM")
    parser.add_option("", "--calibration", type="eng_float", default=0.0,
                      help="set frequency calibration offset to FREQ")
    parser.add_option("-g", "--gain", type="eng_float", default=None,
                      help="set gain in dB (default is midpoint)")
    parser.add_option("-v", "--verbose", action="store_true", default=False,
		      help="print extra debugging info")
    parser.add_option("", "--log-baseband", default=None,
                      help="log filtered baseband to file")
    parser.add_option("", "--log-demod", default=None,
                      help="log demodulator output to file")
    receive_path.add_options(parser, expert_grp)

    for mod in demods.values():
        mod.add_options(expert_grp)

    fusb_options.add_options(expert_grp)
    (options, args) = parser.parse_args ()

    #if len(args) != 0:
    #    parser.print_help(sys.stderr)
    #    sys.exit(1)

    if options.rx_freq is None:
        sys.stderr.write("You must specify -f FREQ or --freq FREQ\n")
        parser.print_help(sys.stderr)
        sys.exit(1)

    access_code=      "10101010101010101110100110001010"
    # build the graph
    tb = my_top_block(demods[options.modulation], rx_callback,access_code, options)

    r = gr.enable_realtime_scheduling()
    if r != gr.RT_OK:
        print "Warning: Failed to enable realtime scheduling."


def main_loop(tb):

    global n_rcvd, n_right,nchan,rfp_pp
    global file_write,print_q,print_n,print_m,print_c,print_p,FILE,FILE1
    print "mainloop started"
    access_code_pp="01010101010101010001011001110101"
    access_code_rfp=      "10101010101010101110100110001010"
    access_code_identity= "10101010101010101110100110001010011"
    access_code=access_code_rfp
    fooz=1
    nchan=1
    rfp_pp=True
    deltaf=0

    print_q=True
    print_n=True
    print_m=True
    print_c=True
    print_p=True
    file_write=False
    

    while fooz in range(0,25):
      #tb.start()
      if (fooz>=10):
        if (fooz==10):
          filename = "adpcm_raw.dat"
          filename1 = "adpcm_raw.txt"
          FILE = open(filename,"wb")
          FILE1 = open(filename1,"w")
          print "data logging to file enabled "
          file_write=True
        if (fooz==11):
          print_q=True
        if (fooz==12):
          print_q=False
        if (fooz==13):
          print_n=True
        if (fooz==14):
          print_n=False
        if (fooz==15):
          print_m=True
        if (fooz==16):
          print_m=False
        if (fooz==17):
          print_c=True
        if (fooz==18):
          print_c=False
        if (fooz==19):
          print_p=True
        if (fooz==20):
          print_p=False
        if (fooz==21):
          access_code=access_code_rfp;
          mazz=tb.rxpath.packet_receiver.set_code(access_code)
          #mazzn=tb.rxpath.packet_receiver.set_code_n(access_code_identity)
          rfp_pp=True
          print "access code set to RFP: ",access_code
        if (fooz==22):
          access_code=access_code_pp;
          mazz=tb.rxpath.packet_receiver.set_code(access_code)
          rfp_pp=False
          print "access code set to PP: ",access_code
        if (fooz==23):
          deltaf=deltaf+5000
          print "deltaf increased by 5 KHz, current value: ",deltaf
        if (fooz==24):
          deltaf=deltaf-5000
          print "deltaf Decreased by 5 KHz, current value: ",deltaf
        #if (fooz==25):
          #tb.rxpath.packet_receiver.pl_me_sync_req()
        #if (fooz==26):
          #print tb.rxpath.packet_receiver.pl_me_sync_ack()



      else:
        n_rcvd = 0
        n_right = 0
        #tb.start()        # start flow graph	
        mazz=tb.rxpath.packet_receiver.set_code(access_code)
        freque=1897344000 - 1728000*fooz
        ok = tb.rxpath.set_freq(freque+deltaf)
        if not ok:
         	print "Failed to set Rx frequency to %s" % (eng_notation.num_to_str(freque+deltaf))
         	raise ValueError, eng_notation.num_to_str(freque+deltaf)
        nchan=fooz
        print "RF frequency = %s" % (eng_notation.num_to_str(freque+deltaf))
        print "CHANNEL = %s" % (nchan)
        #tb.rxpath.packet_receiver.pl_me_sync_req()
        tb.start()
        #print "################################################FLOWGRAPH STOPPED######################################"
      tb.wait()
      fooz=input('Please enter a value between 0 and 9:')
    





if __name__ == '__main__':
  main()
  main_loop(tb)
  #try:
   # main_loop(tb)
  #except KeyboardInterrupt:
    #tb.rxpath.packet_receiver._mac.stop()
    #pass
