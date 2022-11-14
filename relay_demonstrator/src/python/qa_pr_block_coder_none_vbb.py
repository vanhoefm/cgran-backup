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

class qa_pr_block_coder_none_vbb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_block_coder_none_vbb(self):
		frame_size = 155
		num_frames = 16

		data = range(frame_size*num_frames)
		for k in range(len(data)):
			data[k] = data[k] % 2

		expected = data[:]
		src = data[:]

		pseq = range(num_frames)
		exppseq = pseq
		fseq = [0]*num_frames
		expfseq = fseq
		data_valid = [0]*num_frames
		expdata_valid = data_valid

		bsrc0 = qa.txmeta_source(pseq, fseq, data_valid)
		bsrc1 = qa.pvec_source_b(data, frame_size)
		bm = pr.pvec_concat([16, frame_size*gr.sizeof_char])
		uut = pr.block_coder_none_vbb(frame_size)
		be0 = pr.pvec_extract(16+frame_size*gr.sizeof_char, 0, 16)
		be1 = pr.pvec_extract(16+frame_size*gr.sizeof_char, 16, frame_size*gr.sizeof_char)
		bsink0 = qa.txmeta_sink()
		bsink1 = qa.pvec_sink_b(frame_size)

		self.tb.connect(bsrc0, (bm, 0))
		self.tb.connect(bsrc1, (bm, 1))
		self.tb.connect(bm, uut)
		self.tb.connect(uut, be0, bsink0)
		self.tb.connect(uut, be1, bsink1)
		self.tb.run()

		result = bsink1.data()

		self.assertFloatTuplesAlmostEqual(expected, result)
		self.assertFloatTuplesAlmostEqual(bsink0.pkt_seq(), exppseq)
		self.assertFloatTuplesAlmostEqual(bsink0.frame_seq(), expfseq)
		self.assertFloatTuplesAlmostEqual(bsink0.data_valid(), expdata_valid)

if __name__ == '__main__':
	gr_unittest.main()

