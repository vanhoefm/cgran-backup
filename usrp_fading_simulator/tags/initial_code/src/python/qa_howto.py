#!/usr/bin/env python
#
# Copyright 2004 Free Software Foundation, Inc.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from gnuradio import gr, gr_unittest, tait

class qa_tait (gr_unittest.TestCase):

	def setUp (self):
		self.fg = gr.flow_graph ()

	def tearDown (self):
		self.fg = None

	def test_001_square_ff (self):
		src_data = (-3, 4, -5.5, 2, 3)
		expected_result = (9, 16, 30.25, 4, 9)
		src = gr.vector_source_f (src_data)
		sqr = tait.example_ff ()
		dst = gr.vector_sink_f ()
		self.fg.connect (src, sqr)
		self.fg.connect (sqr, dst)
		self.fg.run ()
		result_data = dst.data ()
		self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)
	
	def test_002_biquad4_ss (self):
		src = tait.biquad4_ss()

		
if __name__ == '__main__':
	gr_unittest.main ()
