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

class qa_pr_pvec_extract(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_pvec_extract(self):
		insize = 11
		offset = 4
		outsize = 5

		data = range(5*insize)
		expected = range(4,9) + range(15,20) + range(26,31) + range(37,42) + range(48,53)
		
		bsrc = gr.vector_source_b(data)
		bs2v = pr.stream_to_pvec(gr.sizeof_char, insize)
		uut = pr.pvec_extract(insize, offset, outsize)
		bv2s = pr.pvec_to_stream(gr.sizeof_char, outsize)
		bdst = gr.vector_sink_b()

		self.tb.connect(bsrc, bs2v, uut, bv2s, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

