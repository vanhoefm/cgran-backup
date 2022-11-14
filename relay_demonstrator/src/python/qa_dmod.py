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

from gnuradio import gr, gr_unittest
from openrd import pr, dmod

class qa_dmod(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_dmod(self):
		data = [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]

		src = [0, 0, 1, 1]*16 + data + [0, 0, 1, 1]*16
		
		bsrc = gr.vector_source_b(src)
		mod = dmod.modulator_bc(pr.MODULATION_BPSK)
		tx = dmod.transmitter_cc(pr.MODULATION_BPSK, 4)
		rx = dmod.receiver_cc(pr.MODULATION_BPSK, 4)
		demod = dmod.demodulator_cc(pr.MODULATION_BPSK)
		bdst = gr.vector_sink_c()

		self.tb.connect(bsrc, mod, tx, rx, demod, bdst)
		self.tb.run()

		result = bdst.data()
		
		hd = list(result)
		for k in range(len(hd)):
			if hd[k].real < 0:
				hd[k] = 0
			else:
				hd[k] = 1

		# Check that the source data is contained in the demodulated output
		datafound = 0
		for k in range(len(result)-len(data)):
			if hd[k:k+len(data)] == data:
				datafound = 1

		self.assertTrue(datafound == 1)

if __name__ == '__main__':
	print 'dmod'
	gr_unittest.main()

