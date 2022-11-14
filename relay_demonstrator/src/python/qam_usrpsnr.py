#!/usr/bin/env python
#
# Copyright 2011 Anton Blad.
# 
# This file is part of OpenRD
# 
# OpenRD is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# OpenRD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

"""
USRP SNR measurement test. A complex sinus signal is generated on a USRP 
unit, and the received signal is sampled on another USRP for one second, 
then saved to the specified file for offline analysis.

Two setups are possible:
	1. USRPtx -> USRPrx
	2. USRPtx -> channel -> USRPrx

In the second setup, the channel is provided by qam_usrpsnrch.py. Only the
A sides are used in both setups.

Use the MATLAB script sigsnr.m to compute the SNR of the received signal.
Since complex full-scale signals are used here, the computed SNR is 3 dB
off.
"""

import sys
import time
from gnuradio import gr
from openrd import pyblk 

class usrp_tx(gr.top_block):
	def __init__(self, usrp_serial):
		gr.top_block.__init__(self)
		
		# Sampling rate
		fs = 500000

		# Signal frequency
		freq = 1e3

		# Transmit power
		txpower = -10

		# Blocks
		isrc = gr.sig_source_f (fs, gr.GR_SIN_WAVE, freq, 1.0)
		qsrc = gr.sig_source_f (fs, gr.GR_COS_WAVE, freq, 1.0)
		conv = gr.float_to_complex()
		usrp = pyblk.usrp_c(usrp_serial, 1, 0, fs, txpower)
		
		# Set carrier frequency
		usrp.tune(0)

		# Connect the flow graph
		self.connect(isrc, (conv, 0))
		self.connect(qsrc, (conv, 1))
		self.connect(conv, usrp)

class usrp_rx(gr.top_block):
	def __init__(self, usrp_serial, rxfile):
		gr.top_block.__init__(self)
		
		# Sampling rate
		fs = 500000

		# Receive gain
		rxgain = 10

		# Blocks
		usrp = pyblk.usrp_c(usrp_serial, 0, 1, fs, -10, rxgain)
		head = gr.head(gr.sizeof_gr_complex, fs)
		sink = gr.file_sink(gr.sizeof_gr_complex, rxfile)

		# Set carrier frequency
		usrp.tune(0)

		# Connect the flow graph
		self.connect(usrp, head, sink)

def usage():
	print "usage: qam_usrptx.py usrp_tx_serial usrp_rx_serial rxfile"
	sys.exit(0)

def main():
	if len(sys.argv) < 4:
		usage()
		
	usrp_tx_serial = sys.argv[1]
	usrp_rx_serial = sys.argv[2]
	rxfile = sys.argv[3]
	tx = usrp_tx(usrp_tx_serial)
	rx = usrp_rx(usrp_rx_serial, rxfile)
	tx.start()
	time.sleep(1)
	rx.run()
	tx.stop()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass

