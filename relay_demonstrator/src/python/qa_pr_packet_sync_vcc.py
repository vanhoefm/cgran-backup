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

class qa_pr_packet_sync_vcc(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_packet_sync_vcc(self):
		pkt0 = [0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0]
		pkt1 = [0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0]
		pkt2 = [0,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0]
		pkt3 = [0,0,0,0,1,1,1,0,0,0,1,0,1,0,1,0]
		power = [0]*3
		pseq0 = [0]*3
		pseq1 = [0]*3
		fseq0 = [0, 2, 3]
		fseq1 = [0, 1, 3]
		stamp0 = range(3)
		stamp1 = range(3)
		data0 = pkt0+pkt2+pkt3
		data1 = pkt0+pkt1+pkt3
		frame_size = 16

		expfseq0 = [0, 1, 2, 3]
		expfseq1 = [0, 1, 2, 3]
		expdata0 = pkt0 + [0]*frame_size + pkt2 + pkt3
		expdata1 = pkt0 + pkt1 + [0]*frame_size + pkt3

		bsrc0 = qa.rxframe_source_c(power, stamp0, pseq0, fseq0, [1]*3, data0, frame_size)
		bsrc1 = qa.rxframe_source_c(power, stamp1, pseq1, fseq1, [1]*3, data1, frame_size)
		uut = pr.packet_sync_vcc(frame_size, pr.SEQPOLICY_SYNC, 100)
		bsink0 = qa.rxframe_sink_c(frame_size)
		bsink1 = qa.rxframe_sink_c(frame_size)

		self.tb.connect(bsrc0, (uut, 0))
		self.tb.connect(bsrc1, (uut, 1))
		self.tb.connect((uut, 0), bsink0)
		self.tb.connect((uut, 1), bsink1)
		self.tb.run()

		self.assertFloatTuplesAlmostEqual(expfseq0, bsink0.frame_seq())
		self.assertFloatTuplesAlmostEqual(expfseq1, bsink1.frame_seq())
		self.assertComplexTuplesAlmostEqual(expdata0, bsink0.data())
		self.assertComplexTuplesAlmostEqual(expdata1, bsink1.data())

if __name__ == '__main__':
	gr_unittest.main()

