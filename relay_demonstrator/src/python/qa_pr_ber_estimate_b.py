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
import random

class qa_pr_ber_estimate_b(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()

	def tearDown(self):
		self.tb = None

	def test_001_ber_estimate_b (self):
		l = 16
		alpha = 0.001
		
		beta = 1-alpha

		ref = [0]*l
		src = [0]*l
		for k in range(l):
			src[k] = random.getrandbits(1)

		expber = 0.0
		for k in range(l):
			expber = expber*beta + src[k]*alpha

		bsrc0 = gr.vector_source_b(src)
		bsrc1 = gr.vector_source_b(ref)
		uut = pr.ber_estimate_b(alpha)

		self.tb.connect(bsrc0, (uut, 0))
		self.tb.connect(bsrc1, (uut, 1))

		self.tb.run()

		self.assertAlmostEqual(expber, uut.ber())

if __name__ == '__main__':
	gr_unittest.main ()

