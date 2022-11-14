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
pyblk.usrp_c RX test. The test samples the A side of the USRP with the 
specified serial number for one second, and prints the magnitude average.

Connected to the pyblk.usrp_c TX test, this should give a unit amplitude
when rxgain = -txpower.
"""

import sys
from gnuradio import gr
from openrd import pyblk 

class usrp_rx(gr.top_block):
	def __init__(self, usrp_serial, rxgain):
		gr.top_block.__init__(self)
		
		# Sampling rate
		fs = 1e6

		# Carrier frequency
		freq = 1e3

		# Blocks
		usrp = pyblk.usrp_c(usrp_serial, 0, 1, fs, -10, rxgain)
		c2m = gr.complex_to_mag()
		h = gr.head(gr.sizeof_float, int(fs))
		self.sink = gr.vector_sink_f()
		
		# Set carrier frequency
		usrp.tune(freq)

		# Connect the flow graph
		self.connect(usrp, c2m, h, self.sink)

	def rxmagn(self):
		avg = 0
		for d in self.sink.data():
			avg = avg + d

		return avg / len(self.sink.data())


def usage():
	print "usage: qam_usrprx.py usrp_serial rxgain"
	sys.exit(0)

def main():
	if len(sys.argv) < 3:
		usage()
		
	usrp_serial = sys.argv[1]
	rxgain = int(sys.argv[2])
	rx = usrp_rx(usrp_serial, rxgain)
	rx.run()

	print "Amplitude %f " % rx.rxmagn()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass

