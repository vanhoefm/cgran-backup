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

class qa_pr_frame_sync_cc(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()

	def tearDown(self):
		self.tb = None

	def test_001_frame_sync_cc (self):
		frame_size = 10
		src = range(48)
		sync = [0]*5 + [1] + [0]*9 + [1] + [0]*12 + [1] + [0]*19
		exppwr = [0]*3
		exptype = [1]*3
		expdata = range(5,15) + range(15,25) + range(28,38)

		bsrc = gr.vector_source_c(src)
		bsync = gr.vector_source_i(sync)
		uut = pr.frame_sync_cc(frame_size)
		bsink = qa.rxframe_sink_c(frame_size)

		self.tb.connect(bsrc, (uut, 0))
		self.tb.connect(bsync, (uut, 1))
		self.tb.connect(uut, bsink)

		self.tb.run()

		self.assertFloatTuplesAlmostEqual(exppwr, bsink.power())
		self.assertFloatTuplesAlmostEqual(exptype, bsink.frame_type())
		self.assertComplexTuplesAlmostEqual(expdata, bsink.data())

if __name__ == '__main__':
	gr_unittest.main ()
