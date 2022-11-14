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

class qa_pr_block_decoder_none_vfb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_block_decoder_none_vfb(self):
		frame_size = 155
		num_frames = 16

		pkt_seq = range(num_frames)
		data = range(frame_size*num_frames)
		for k in range(len(data)):
			data[k] = ((data[k]*31 % 256)-128)*0.1

		expected = data[:]

		for k in range(len(expected)):
			if expected[k] >= 0.0:
				expected[k] = 1
			else:
				expected[k] = 0

		src = data[:]

		bsrc = qa.rxmeta_source_f(pkt_seq, [0]*num_frames, src, frame_size)
		uut = pr.block_decoder_none_vfb(frame_size)
		bsink = qa.rxmeta_sink_b(frame_size)

		self.tb.connect(bsrc, uut, bsink)
		self.tb.run()

		respseq = bsink.pkt_seq()
		result = bsink.data()

		self.assertFloatTuplesAlmostEqual(pkt_seq, respseq)
		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

