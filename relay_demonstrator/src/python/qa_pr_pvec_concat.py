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
from openrd import pr

class qa_pr_pvec_concat(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_pvec_concat(self):
		num0 = 3
		num1 = 2
		size0 = gr.sizeof_char*num0
		size1 = gr.sizeof_float*num1
		src0 = range(9)
		src1 = range(6)

		expected0 = src0
		expected1 = src1
		
		bsrc0 = gr.vector_source_b(src0)
		bsrc1 = gr.vector_source_f(src1)
		bs2v0 = pr.stream_to_pvec(gr.sizeof_char, num0)
		bs2v1 = pr.stream_to_pvec(gr.sizeof_float, num1)
		uut = pr.pvec_concat([size0, size1])
		be0 = pr.pvec_extract(size0+size1, 0, size0)
		be1 = pr.pvec_extract(size0+size1, size0, size1)
		bv2s0 = pr.pvec_to_stream(gr.sizeof_char, num0)
		bv2s1 = pr.pvec_to_stream(gr.sizeof_float, num1)
		bdst0 = gr.vector_sink_b()
		bdst1 = gr.vector_sink_f()

		self.tb.connect(bsrc0, bs2v0, (uut, 0))
		self.tb.connect(bsrc1, bs2v1, (uut, 1))
		self.tb.connect(uut, be0, bv2s0, bdst0)
		self.tb.connect(uut, be1, bv2s1, bdst1)
		self.tb.run()

		result0 = bdst0.data()
		result1 = bdst1.data()

		self.assertFloatTuplesAlmostEqual(expected0, result0)
		self.assertFloatTuplesAlmostEqual(expected1, result1)

if __name__ == '__main__':
	gr_unittest.main()

