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

import struct,sys
import gnuradio.gr.gr_threading as _threading

VERBOSE = 0

class Send_thread(_threading.Thread):
    def __init__(self, global_ctrl, send_pkt, send_data):
        _threading.Thread.__init__(self)
        self.global_ctrl = global_ctrl
        self.send_pkt = send_pkt
        self.setDaemon(1)
        self.send_data = send_data
    
    def run(self):
      ###--- Transmitter - send data ---###
      options = self.global_ctrl.get_options()
      if options.benchmark_mode:
          # generate and send packets
          npixels = options.npixels
          n = 0
          s_pktno = 0
          color = 0
          pkt_size = int(options.size)-8
          while n < npixels:
              if npixels-n < pkt_size:
                  pkt_size = int(npixels-n)
              #pkt_contents = struct.pack('!H', s_pktno) + (pkt_size - 2) * (chr(0 & 0xff) + chr(color & 0xff) + chr(0 & 0xff)) #green colors...
              pkt_contents = pkt_size/3 * (chr(0 & 0xff) + chr(color & 0xff) + chr(0 & 0xff)) #green colors...
              self.send_pkt(pkt_contents)
              sys.stdout.write(".")
              n += pkt_size
              s_pktno += 1
              color += 1
              if color==(256): color=0
              
          self.send_pkt(eof=True)
          print "\nThe transmitter program sent: ", s_pktno-1, "OFDM packets"
      elif options.file_mode:
          # Send packets
          pkt_size = int(options.size)-8
          n_sent = 0
          # Read input data from file:
          in_file = open(options.input_filename,mode="rb")
          indata = in_file.read(pkt_size)
          read_pkt_size = len(indata)
          if VERBOSE:
              print "TB: Reading first pkt of size",read_pkt_size
          while (indata):
              #if VERBOSE:
              #           print "Last byte:",indata[read_pkt_size-1]
              self.send_pkt(indata)
              n_sent += 1
              if VERBOSE:
                  print "TB: Sent pkt!"
              indata = in_file.read(pkt_size)
              read_pkt_size = len(indata)
              if VERBOSE:
                  print "TB: Reading another pkt of size",read_pkt_size
          else:
              self.send_pkt(indata)        # Sent one empty pkt to get last pkt... bug?
              self.send_pkt(eof=True)
              if VERBOSE:
                  print "Sent the last empty packet and eof"
          in_file.close()
          print "Sent",n_sent,"packets."
      elif options.image_mode:
          pkt_size = int(options.size)-8        #Remember to send a multiple of 3 to get one pixel
          n_sent = 0
          indata = self.send_data
          tot_size = len(indata)
          nr_pkts = tot_size/pkt_size # rounds by automatic int/int
          if(tot_size%pkt_size): nr_pkts+=1
          for i in range(0,nr_pkts):                                     #OBS range(0,i) gives (0,...,i-1)
              if(n_sent == nr_pkts-1):
                  self.send_pkt(indata[i*pkt_size:])
              else:
                  self.send_pkt(indata[i*pkt_size:(i+1)*pkt_size])       #OBS list[i:j] gives [i,i+1,...,j-1]
              n_sent += 1
              if VERBOSE:
                  print "TB: Sent pkt!"
          self.send_pkt("")        # Sent one empty pkt to get last pkt... bug?
          self.send_pkt(eof=True)
          if VERBOSE:
              print "Sent the last empty packet and eof"
          print "Sent",n_sent,"packets."
      elif options.ber_mode:
        # generate and send packets
        nbytes = options.npixels
        n = 0
        s_pktno = 0
        pkt_size = int(options.size)-8  
        while n < nbytes:
            pkt_contents = struct.pack('!H', s_pktno) + (pkt_size - 2) * chr(0 & 0xff)
        
            self.send_pkt(pkt_contents)
            #sys.stdout.write(".")
            n += pkt_size
            s_pktno += 1
        
        self.send_pkt(eof=True)
        print "\nThe transmitter program sent: ", s_pktno-1, "OFDM packets"  
      
      #self.global_ctrl.stop_flowgraph() #This did not work very well, especially not in benchmark mode
