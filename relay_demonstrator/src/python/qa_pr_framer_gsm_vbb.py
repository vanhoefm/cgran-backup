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

class qa_pr_framer_gsm_vbb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_framer_gsm_vbb(self):
		num_frames = 3
		frame_size = 142
		synccode = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
		datacode = [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]

		data_size = frame_size-len(datacode)

		data1 = range(data_size)
		data2 = range(8,8+data_size)
		data3 = range(24,24+data_size)

		expsync = [0]*39 + synccode + [0]*39
		expd1 = data1[0:57] + datacode + data1[57:]
		expd2 = data2[0:57] + datacode + data2[57:]
		expd3 = data3[0:57] + datacode + data3[57:]

		src = data1 + data2 + data3
		expected = expsync + expd1 + expd2 + expd3

		pseq = [0]*num_frames
		exppseq = [0]+pseq
		fseq = range(num_frames)
		expfseq = [0]+fseq
		data_valid = [0]*num_frames
		expdata_valid = [0]+data_valid

		bsrc0 = qa.txmeta_source(pseq, fseq, data_valid)
		bsrc1 = qa.pvec_source_b(src, data_size)
		bm = pr.pvec_concat([16, data_size])
		uut = pr.framer_gsm_vbb(frame_size, pr.FIELD_CODE_32_78, synccode, datacode)
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

