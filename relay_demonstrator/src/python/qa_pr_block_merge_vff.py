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

class qa_pr_block_merge_vff(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_block_merge_vff(self):
		insize = 30
		outsize = 150
		f = outsize/insize
		num = 3

		power = [0]*(f*num) + [0]
		pseq = [0]*f + [1]*f + [2]*f + [3]
		fseq = range(f)*num + [0]
		stamp = range(len(power))

		expected = range(outsize)*num
		data = expected[:] + [0]*insize

		bsrc = qa.rxframe_source_f(power, stamp, pseq, fseq, [1]*(num*f+1), data, insize)
		uut = pr.block_merge_vff(insize, outsize)
		bsink = qa.rxmeta_sink_f(outsize)

		self.tb.connect(bsrc, uut, bsink)
		self.tb.run()

		resseq = bsink.pkt_seq()
		result = bsink.data()

		self.assertFloatTuplesAlmostEqual(expected, result)
		self.assertFloatTuplesAlmostEqual(range(num), resseq)

	def test_002_block_merge_vff(self):

		return
		insize = 3
		outsize = 9

		data = [1, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0, 1, 9, 10, 11, 1, 12, 13, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0]
		expected = range(18)
		
		bsrc = gr.vector_source_f(data)
		bs2v = pr.stream_to_pvec(gr.sizeof_float, insize+1)
		uut = pr.block_merge_vff(insize, outsize)
		bv2s = pr.pvec_to_stream(gr.sizeof_float, outsize)
		bdst = gr.vector_sink_f()

		self.tb.connect(bsrc, bs2v, uut, bv2s, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

