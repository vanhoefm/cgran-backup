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

class qa_pr_data_source_counter(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_data_source_counter(self):
		frame_size = 12
		num_frames = 16

		expected = [0,0,0,0,0,0,0,1,0,0,1,0]*num_frames

		uut = pr.data_source_counter(frame_size)
		bhead = gr.head(pr.pvec_alloc_size(16+frame_size), num_frames)
		bdst = qa.txmeta_sink_b(frame_size)

		self.tb.connect(uut, bhead, bdst)
		self.tb.run()

		self.assertFloatTuplesAlmostEqual(range(num_frames), bdst.pkt_seq())
		self.assertFloatTuplesAlmostEqual([0]*num_frames, bdst.frame_seq())
		self.assertFloatTuplesAlmostEqual([1]*num_frames, bdst.data_valid())
		self.assertFloatTuplesAlmostEqual(expected, bdst.data())

if __name__ == '__main__':
	gr_unittest.main()

