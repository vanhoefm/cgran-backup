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
pyblk.usrp_c TX test. The test should generate a sinus signal on the A side 
of the USRP with the specified serial number.
"""

import sys
from gnuradio import gr
from openrd import pyblk 

class usrp_tx(gr.top_block):
	def __init__(self, usrp_serial, txpower):
		gr.top_block.__init__(self)
		
		# Sampling rate
		fs = 1e6

		# Carrier frequency
		freq = 1e3

		# Blocks
		src = gr.sig_source_f (fs, gr.GR_COS_WAVE, 0, 1.0)
		conv = gr.float_to_complex()
		usrp = pyblk.usrp_c(usrp_serial, 1, 0, fs, txpower)
		
		# Set carrier frequency
		usrp.tune(freq)

		# Connect the flow graph
		self.connect(src, conv, usrp)

def usage():
	print "usage: qam_usrptx.py usrp_serial txpower"
	sys.exit(0)

def main():
	if len(sys.argv) < 3:
		usage()
		
	usrp_serial = sys.argv[1]
	txpower = int(sys.argv[2])
	tx = usrp_tx(usrp_serial, txpower)
	tx.run()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass

