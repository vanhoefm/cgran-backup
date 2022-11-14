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

class qa_analyzer_ber_vb (gr_unittest.TestCase):

	def setUp (self):
		self.tb = gr.top_block ()

	def tearDown (self):
		self.tb = None

	def test_001_analyzer_ber_vb (self):
		block_size = 1024
		num_blocks = 8

		stream1 = [1]*256 + [0]*128 + [1]*512 + [0]*128 + [1]*3072 + [1]*256 + [1]*128 + [1]*512 + [0]*128 + [0]*3072
		stream2 = [0]*256 + [0]*128 + [1]*512 + [0]*128 + [1]*3072 + [1]*256 + [0]*128 + [1]*512 + [0]*128 + [0]*3072
		
		expected_result_ber = [256.0/2048, 0.0/2048, 128.0/2048, 0.0/2048]
		expected_result_bler = [0.5, 0, 0.5, 0]

		src1 = qa.txmeta_source_b(range(num_blocks), [0]*num_blocks, [1]*num_blocks, stream1, block_size)
		src2 = qa.rxmeta_source_b(range(num_blocks), [0]*num_blocks, stream2, block_size)

		uut = pr.analyzer_ber_vb(block_size, 2)
		
		self.tb.connect(src1, (uut, 0))
		self.tb.connect(src2, (uut, 1))
		self.tb.run()

		self.assertComplexTuplesAlmostEqual(expected_result_ber, uut.ber())
		self.assertComplexTuplesAlmostEqual(expected_result_bler, uut.bler())

if __name__ == '__main__':
	gr_unittest.main ()

