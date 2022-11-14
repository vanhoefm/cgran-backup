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

class qa_pr_stream_to_pvec(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_stream_to_pvec(self):
		size = 11
		block = 16

		data = range(size) + [0]*(block-size) + range(2,2+size) + [0]*(block-size)
		expected = range(size) + range(2,2+size)
		
		bsrc = gr.vector_source_b(data)
		bs2v = gr.stream_to_vector(gr.sizeof_char, block)
		uut = pr.pvec_to_stream(gr.sizeof_char, size)
		bdst = gr.vector_sink_b()

		self.tb.connect(bsrc, bs2v, uut, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

