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
from openrd import pr

class qa_pr_rate_estimate(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_rate_estimate(self):
		size = 1000000

		data = [0]*size
		
		bsrc = gr.vector_source_b(data)
		bthr = gr.throttle(gr.sizeof_char, size)
		uut = pr.rate_estimate(gr.sizeof_char)

		self.tb.connect(bsrc, bthr, uut)
		self.tb.run()

		#print "Rate %f" % uut.rate()
		
		diff = uut.rate()-size
		if diff < 0.0:
			diff = -diff

		self.assertTrue(diff/size < 0.1)

if __name__ == '__main__':
	gr_unittest.main()

