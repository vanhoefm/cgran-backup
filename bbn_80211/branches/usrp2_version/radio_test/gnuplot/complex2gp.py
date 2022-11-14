#!/usr/bin/env python

#  Copyright 2005 BBN Technologies. All rights reserved.
#  Description: 
#	This file implements near-real-time graphig capabilities.
#	It reads from the "decoded_rx.dat' file procuded by phy_ll.
#
# $Header: /cvs/adroitgrdevel/adroitgrdevel/radio_test/gnuplot/complex2gp.py,v 1.1 2006/05/19 20:31:59 bvincent Exp $
#
# $Author: bvincent $
#
# $Revision: 1.1 $
#
# $Date: 2006/05/19 20:31:59 $
#

# For oper interface.
import sys
import cmd
import array
import struct

# For graph.
# import os
import socket
import time
#from gnuradio import eng_notation
#from gnuradio.eng_option import eng_option
from optparse import OptionParser

# Here is what people have been up to:
#
# $Log: complex2gp.py,v $
# Revision 1.1  2006/05/19 20:31:59  bvincent
# Conversion utility to plot raw data in gnuplot.
#


def plot_stuff ():
	self._myname = socket.getfqdn (socket.gethostname ())
	cmd = "gnuplot"
	( gnuplot, gnuplot_stdout, gnuplot_stderr) = os.popen3 (cmd, "t", -1)
	plotcmd = "set grid\n"
	gnuplot.write (plotcmd)
	plotcmd = "set ticslevel 0\n"
	gnuplot.write (plotcmd)
	plotcmd = "set title \'Frame Samples for " + self._myname + "\'\n"
	gnuplot.write (plotcmd)
	plotcmd = "set xlabel \'Frame Number\'\n"
	gnuplot.write (plotcmd)
	plotcmd = "set ylabel \'Frame Sample Number\'\n"
	gnuplot.write (plotcmd)
	plotcmd = "set zlabel \'Power\'\n"
	gnuplot.write (plotcmd)
	plotcmd = "set view 30,40\n"
	gnuplot.write (plotcmd)
	if self._myname == "lagrange.bbn.com":
		plotcmd = "set zrange [0:7e6]\n"
	elif self._myname == "maxwell.bbn.com":
		plotcmd = "set zrange [0:1.2e7]\n"
	gnuplot.write (plotcmd)
	plotcmd = "splot "
 	plotcmd += "\'raw_rx_disc_scale.dat\' using 1:2:3 with impulse t \'\', "
	plotcmd += "\'raw_rx_synch_scale.dat\' using 1:2:3 with impulse t \'\', "
	plotcmd += "\'raw_rx.dat\' using 1:2:3 with impulse t \'Received power\'\n"
        # plotcmd += "\'raw_rx.dat\' using 1:2:4 with impulse\n"
	gnuplot.write (plotcmd)
	gnuplot.flush ()
	while self._running:
		time.sleep (5.0)
		gnuplot.write (plotcmd)
		gnuplot.flush ()
		
		gnuplot.close ()

def proc (log_file_name, dat_file_name):
    rcv_log_file = open (log_file_name, 'r')
    rcv_dat_file = open (dat_file_name, 'w')

    frame_size_b = 8
    frame_number = 0
    while True:
        # print ('Processing frame: %d'%(frame_number,))
        in_string = rcv_log_file.read (frame_size_b)
        if not in_string: break
        if len (in_string) < frame_size_b:
            print ('Read in %d not %d bytes', len(in_string), frame_size_s)
        offset = 0
        for s in xrange (0,len(in_string),8):
            # angle = math.atan2 (imag, real)
	    imag, real = struct.unpack ('ff', in_string[s:s+8])
	    phasor = complex (real, imag)
	    power = abs (phasor)**2
	    
	    #print real, imag, phasor, power
	    rcv_dat_file.write ('%d %f %f %f \n' %(frame_number, real, imag, power))
            offset += 1
        frame_number += 1

    rcv_log_file.close ()
    rcv_dat_file.close ()

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-i", "--in_filename", type="string", default="in.dat",
                          help="input filename to process")
	parser.add_option("-o", "--out_filename", type="string", default="out.dat",
                          help="output filename")

        (options, args) = parser.parse_args()
        if len(args) != 0:
            parser.print_help()
            sys.exit(1)

	# Do stuff.
	proc (options.in_filename, options.out_filename )

