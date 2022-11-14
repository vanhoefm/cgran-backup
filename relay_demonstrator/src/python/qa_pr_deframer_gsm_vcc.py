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

class qa_pr_deframer_gsm_vcc(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()

	def tearDown(self):
		self.tb = None

	def test_001_deframer_gsm_vcc (self):
		scode = [1,-1,1,-1,1,1,-1,-1]
		dcode = [1,1,1,1,1,1,1,1]
		frame_size = 39+8+39
		data_size = 39*2
		block = 39
		srcpwr = [0]*3
		srcstamp = range(3)
		srctype = [pr.GSM_FRAME_SYNC, pr.GSM_FRAME_DATA, pr.GSM_FRAME_DATA+1]
		srcdata = [-1]*block + scode + [-1]*block + range(block) + dcode + range(block) + range(block) + dcode + range(block)

		exppwr = [0,0]
		exppseq = [0,0]
		expfseq = [0,1]
		expdata = range(block)*4

		bsrc = qa.rxframe_source_c(srcpwr, srcstamp, [0]*3, [0]*3, srctype, srcdata, frame_size)
		uut = pr.deframer_gsm_vcc(frame_size, pr.FIELD_CODE_32_78, scode, dcode)
		bsink = qa.rxframe_sink_c(data_size)

		self.tb.connect(bsrc, uut, bsink)
		self.tb.run()

		#self.assertFloatTuplesAlmostEqual(respwr, exppwr)
		self.assertFloatTuplesAlmostEqual(exppseq, bsink.pkt_seq())
		self.assertFloatTuplesAlmostEqual(expfseq, bsink.frame_seq())
		self.assertComplexTuplesAlmostEqual(expdata, bsink.data())

if __name__ == '__main__':
	gr_unittest.main ()
