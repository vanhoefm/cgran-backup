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

class qa_pr_frame_correlator_simple_bb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_frame_correlator_simple_bb(self):
		code = [1, 0, 1, 0, 1, 1, 0, 0]
		frame_size = 15
		input_size = 1
		data = [0]*5 + code + range(7) + [1]*9 + code + range(7) + [0]*4
		expected = [0]*19 + [1] + [0]*23 + [1] + [0]*4

		src = data[:]

		bsrc = gr.vector_source_b(src)
		bs2v = pr.stream_to_pvec(gr.sizeof_char, input_size)
		uut = pr.frame_correlator_simple_bb(input_size, frame_size, code, 8)
		bdst = gr.vector_sink_i()

		self.tb.connect(bsrc, bs2v, uut, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

	def test_002_frame_correlator_simple_bb(self):
		code = [1, 0, 1, 0, 0, 1, 0, 0]
		frame_size = 12
		input_size = 2
		data = [0]*10 + code + range(4) + [1]*6 + code + range(4) + [1]*4

		src = data[:]
		expected = [0]*10 + [1] + [0]*8 + [1] + [0]*2

		bsrc = gr.vector_source_b(src)
		bs2v = pr.stream_to_pvec(gr.sizeof_char, input_size)
		uut = pr.frame_correlator_simple_bb(input_size, frame_size, code, 7)
		bdst = gr.vector_sink_i()

		self.tb.connect(bsrc, bs2v, uut, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

