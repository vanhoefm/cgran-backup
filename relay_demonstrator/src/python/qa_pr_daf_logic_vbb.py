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

class qa_pr_daf_logic_vbb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()

	def tearDown(self):
		self.tb = None

	def test_001_daf_logic_vbb (self):
		frame_size = 155
		num_frames = 16
		
		data = range(frame_size*num_frames)
		for k in range(len(data)):
			data[k] = data[k] % 256

		expdata = data[:]
		srcdata = data[:]

		pseq = range(num_frames)
		decoded = [0]*num_frames

		exppseq = pseq
		expfseq = [0]*num_frames
		expdata_valid = [1]*num_frames

		bsrc = qa.rxmeta_source_b(pseq, decoded, data, frame_size)
		uut = pr.daf_logic_vbb(frame_size)
		bsink = qa.txmeta_sink_b(frame_size)

		self.tb.connect(bsrc, uut, bsink)
		self.tb.run()

		self.assertFloatTuplesAlmostEqual(exppseq, bsink.pkt_seq())
		self.assertFloatTuplesAlmostEqual(expfseq, bsink.frame_seq())
		self.assertFloatTuplesAlmostEqual(expdata_valid, bsink.data_valid())
		self.assertComplexTuplesAlmostEqual(expdata, bsink.data())

if __name__ == '__main__':
	gr_unittest.main ()
