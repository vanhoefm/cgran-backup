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

import sys
import time
from gnuradio import gr
from openrd import pr, pyblk, channel

def ber(ref, x):
	num = 0

	for k in range(len(ref)):
		if ref[k] != x[k]:
			num = num + 1
	
	return num

def bitdiff(ref, x):
	d = []

	for k in range(len(ref)):
		if ref[k] != x[k]:
			d.append(k)
	
	return d

class bpsksnr(gr.top_block):
	def __init__(self):
		gr.top_block.__init__(self)

		data = [0,1]*1024 + [0,0,1,1]*512 + [0]*128 + [1]*128
		#data = [0]*4096
		data = data * 160
		src = [0]*128 + [0,1]*128 + data + [0,1]*128 + [0]*128

		bsrc = gr.vector_source_b(src)

		mod = pyblk.modulator_bc(pr.MODULATION_BPSK, 4)

		chn = channel.channel_awgn(pr.MODULATION_BPSK, -10)

		demod = pyblk.demodulator_cc(pr.MODULATION_BPSK, 4)

		c2r = gr.complex_to_real()
		slice = gr.binary_slicer_fb()

		bsink = gr.vector_sink_b()

		self.connect(mod, gr.file_sink(8, 'mod'))
		self.connect(chn, gr.file_sink(8, 'dmod0'))

		self.connect(bsrc, mod, chn, demod, c2r, slice, bsink)
		self.run()

		minber = len(data)
		minofs = 0
		#for ofs in range(256,512):
		for ofs in range(400,420):
			b = ber(data, bsink.data()[ofs:ofs+len(data)])
			if b < minber:
				minber = b
				minofs = ofs

		#print bitdiff(data, bsink.data()[minofs:minofs+len(data)])

		print len(bsink.data())
		print minofs
		print minber
		print float(minber)/float(len(data))

	
def main():
	tb = bpsksnr()

main()

