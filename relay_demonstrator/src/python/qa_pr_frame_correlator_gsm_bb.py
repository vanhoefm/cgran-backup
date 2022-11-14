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
from openrd import pr, pr

class qa_pr_frame_correlator_gsm_bb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_frame_correlator_gsm_bb(self):
		scode = [1, 0, 1, 0, 1, 1, 0, 0]
		dcode = [1, 1, 1, 1, 1, 1, 1, 1]
		frame_size = 16
		input_size = 1
		data = [0]*4 + scode + [0]*4 + range(4) + dcode + range(4) + range(4) + dcode + range(4)
		expected = [0]*15 + [pr.GSM_FRAME_SYNC] + [0]*15 + [pr.GSM_FRAME_DATA] + [0]*15 + [pr.GSM_FRAME_DATA+1]

		src = data[:]

		bsrc = gr.vector_source_b(src)
		bs2v = pr.stream_to_pvec(gr.sizeof_char, input_size)
		uut = pr.frame_correlator_gsm_bb(input_size, frame_size, scode, 8, dcode, 8)
		bdst = gr.vector_sink_i()

		self.tb.connect(bsrc, bs2v, uut, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

