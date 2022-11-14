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

class qa_pr_mrc_vcc(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_mrc_vcc(self):
		size = 5

		power = [1, 1]
		pseq = [0, 0]
		fseq = [0, 1]
		stamp = [0, 1]
		src0 = [1+2j, 2, 3+1j, 0, 0, 2j, 1+1j, 1+2j, 0, 0];
		src1 = [0, 1+2j, 2, 3+1j, 0, 0, 2j, 1+1j, 1+2j, 0];
		expected = [0]*len(src0)
		for k in range(len(src0)):
			expected[k] = src0[k]+src1[k]

		bsrc0 = qa.rxframe_source_c(power, stamp, pseq, fseq, [0]*2, src0, size)
		bsrc1 = qa.rxframe_source_c(power, stamp, pseq, fseq, [0]*2, src1, size)
		uut = pr.mrc_vcc(size)
		bsink = qa.rxframe_sink_c(size)

		self.tb.connect(bsrc0, (uut, 0))
		self.tb.connect(bsrc1, (uut, 1))
		self.tb.connect(uut, bsink)
		self.tb.run()

		self.assertFloatTuplesAlmostEqual(pseq, bsink.pkt_seq())
		self.assertFloatTuplesAlmostEqual(fseq, bsink.frame_seq())
		self.assertComplexTuplesAlmostEqual(expected, bsink.data())

if __name__ == '__main__':
	gr_unittest.main()

