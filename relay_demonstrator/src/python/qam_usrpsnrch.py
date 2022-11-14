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
Channel for USRP SNR measurement test.

Use together with qam_usrpsnr.py
"""

import sys
import time
from gnuradio import gr
from openrd import pyblk, channel

class usrp_channel(gr.top_block):
	def __init__(self, usrp_serial, ch):
		gr.top_block.__init__(self)
		
		# Sampling rate
		fs = 500000

		# Transmit power
		txpower = -10

		# Receive gain
		rxgain = 10

		# Blocks
		usrp = pyblk.usrp_c(usrp_serial, 1, 1, fs, txpower, rxgain)
		
		# Set carrier frequency
		usrp.tune(0)

		# Connect the flow graph
		self.connect(usrp, ch)
		self.connect(ch, usrp)

def usage():
	print "usage: qam_usrpsnrch.py usrp_serial \"i\""
	print "         using ideal channel"
	print "       qam_usrpsnrch.py usrp_serial \"a\" dbfs"
	print "         using awgn channel"
	print "       qam_usrpsnrch.py usrp_serial \"r\" speed dbfs"
	print "         using Rayleigh channel"
	sys.exit(0)

def main():
	if len(sys.argv) < 3:
		usage()

	usrp_serial = sys.argv[1]
	mode = sys.argv[2]

	if mode == 'i':
		chan = channel.channel_ideal()
	elif mode == 'a':
		chan = channel.channel_awgn(int(sys.argv[3]))
	else:
		usage()

	ch = usrp_channel(usrp_serial, chan)
	ch.run()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass

