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

class qa_pr_packet(gr_unittest.TestCase):
	def setUp(self):
		pass
	
	def tearDown(self):
		pass
	
	def test_001_packet(self):
		size = 12
		
		p1 = pr.dpacket(size)

		p1.set_protocol(3)
		p1.set_initial_packet()
		p1.set_length(42)
		p1.set_seq(12)
		p1.set_datav(range(p1.data_size()))
		p1.calculate_crc()

		p2 = pr.dpacket(size)
		p2.set_rawv(p1.rawv())

		self.assertTrue(p2.check_crc())
		self.assertEqual(p1.protocol(), p2.protocol())
		self.assertEqual(p1.initial_packet(), p2.initial_packet())
		self.assertEqual(p1.length(), p2.length())
		self.assertEqual(p1.datav(), p2.datav())
		self.assertEqual(p1.protocol(), 3)
		self.assertEqual(p1.initial_packet(), True)
		self.assertEqual(p1.length(), 42)
		self.assertFloatTuplesAlmostEqual(p1.datav(), range(p1.data_size()))

if __name__ == '__main__':
	gr_unittest.main()

