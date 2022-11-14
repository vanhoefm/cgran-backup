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

class qa_pr_block_partition_vbb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_block_partition_vbb(self):
		insize = 150
		outsize = 30
		num = 3
		d_factor = insize/outsize

		data = range(insize)*num
		expected = data[:]

		pseq = range(num)
		fseq = [0]*num
		expfseq = range(d_factor)*num
		data_valid = [0]*num
		expdata_valid = [0]*d_factor*num
		
		bsrc0 = qa.txmeta_source(pseq, fseq, data_valid)
		bsrc1 = qa.pvec_source_b(data, insize)
		bm = pr.pvec_concat([16, insize*gr.sizeof_char])
		uut = pr.block_partition_vbb(insize, outsize)
		be0 = pr.pvec_extract(16+outsize, 0, 16)
		be1 = pr.pvec_extract(16+outsize, 16, outsize)
		bsink0 = qa.txmeta_sink()
		bsink1 = qa.pvec_sink_b(outsize)

		self.tb.connect(bsrc0, (bm, 0))
		self.tb.connect(bsrc1, (bm, 1))
		self.tb.connect(bm, uut)
		self.tb.connect(uut, be0, bsink0)
		self.tb.connect(uut, be1, bsink1)
		self.tb.run()

		result = bsink1.data()

		self.assertFloatTuplesAlmostEqual(expected, result)
		self.assertFloatTuplesAlmostEqual(bsink0.pkt_seq(), [0]*5 + [1]*5 + [2]*5)
		self.assertFloatTuplesAlmostEqual(bsink0.frame_seq(), range(5)*num)
		self.assertFloatTuplesAlmostEqual(bsink0.data_valid(), expdata_valid)

if __name__ == '__main__':
	gr_unittest.main()

