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
from openrd import pr, pr, qa

class qa_pr_const_mapper_vbc(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_const_mapper_vbc(self):
		mod = pr.MODULATION_BPSK

		frame_size = 155
		num_frames = 16

		data = range(frame_size*num_frames)
		for k in range(len(data)):
			data[k] = data[k] % 2

		src = data[:]
		expected = data[:]

		for k in range(len(expected)):
			if expected[k] == 0:
				expected[k] = -1
			else:
				expected[k] = 1
		
		expected = expected[0:frame_size*10] + [0]*frame_size + expected[frame_size*11:frame_size*num_frames]
		
		pseq = [0]*num_frames
		exppseq = pseq
		fseq = range(num_frames)
		expfseq = range(10) + [0] + range(11,16)
		data_valid = [1]*10 + [0] + [1]*5
		expdata_valid = data_valid

		bsrc0 = qa.txmeta_source(pseq, fseq, data_valid)
		bsrc1 = qa.pvec_source_b(data, frame_size)
		bm = pr.pvec_concat([16, frame_size*gr.sizeof_char])
		uut = pr.const_mapper_vbc(frame_size, mod)
		bsink1 = qa.pvec_sink_c(frame_size)

		self.tb.connect(bsrc0, (bm, 0))
		self.tb.connect(bsrc1, (bm, 1))
		self.tb.connect(bm, uut)
		self.tb.connect(uut, bsink1)
		self.tb.run()

		result = bsink1.data()

		self.assertFloatTuplesAlmostEqual(expected, result)


if __name__ == '__main__':
	gr_unittest.main()

