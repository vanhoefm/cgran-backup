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
from openrd import pr, qa

class qa_pr_constellation_softdecoder_vcf(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_constellation_softdecoder_vcf(self):
		mod = pr.MODULATION_BPSK
		
		frame_size = 3

		power = [0, 0]
		pseq = [0, 0]
		fseq = [0, 1]
		stamp = [0, 1]
		data = (-0.5, 0.3+1j, 0.7-7j, 1.2, -4.8+2j, 0.8+1j)
		expected = (-0.5, 0.3, 0.7, 1.2, -4.8, 0.8)
		
		bsrc = qa.rxframe_source_c(power, stamp, pseq, fseq, [0]*2, data, frame_size)
		uut = pr.constellation_softdecoder_vcf(mod, frame_size)
		bsink = qa.rxframe_sink_f(frame_size)

		self.tb.connect(bsrc, uut, bsink)
		self.tb.run()

		result = bsink.data()

		self.assertFloatTuplesAlmostEqual(expected, result, 5)

if __name__ == '__main__':
	gr_unittest.main()

