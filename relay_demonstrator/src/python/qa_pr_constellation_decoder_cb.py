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
from openrd import pr, pr

class qa_pr_constellation_decoder_cb(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_constellation_decoder_cb(self):
		mod = pr.MODULATION_BPSK
		
		data = (-0.5, 0.3+1j, 0.7-7j, 1.2, -4.8+2j)
		expected = (0, 1, 1, 1, 0)
		
		bsrc = gr.vector_source_c(data)
		uut = pr.constellation_decoder_cb(mod)
		bv2s = pr.pvec_to_stream(gr.sizeof_char, pr.bits_per_symbol(mod))
		bdst = gr.vector_sink_b()

		self.tb.connect(bsrc, uut, bv2s, bdst)
		self.tb.run()

		result = bdst.data()

		self.assertFloatTuplesAlmostEqual(expected, result)

if __name__ == '__main__':
	gr_unittest.main()

