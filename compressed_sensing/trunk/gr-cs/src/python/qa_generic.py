#!/usr/bin/env python
#
# Copyright 2001-2008 Free Software Foundation, Inc.
# Copyright 2009 Institut fuer Nachrichtentechnik / Uni Karlsruhe
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from gnuradio import gr, gr_unittest
import cs

from numpy import *

class qa_cs_generic(gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None
	self.cs_f = None

    def test_001_deterministic (self):
	input_data = (0+0j, 1+0j, 2+0j, 0+0j, 1+0j, 2+0j)

	M = 2
	N = 3

	comp_matrix_f = ((1.0, 1.0, 0.0), (0.0, 2.5, 3.0))
	#comp_matrix_s = ((1, 1, 0), (0, 2, 3))
	self.cs_f = cs.generic_vccf(comp_matrix_f)
	#self.cs_s = cs.generic_vccf(self.comp_matrix_s)
	src = gr.vector_source_c(input_data, False, N)
	sink_f = gr.vector_sink_c(M)
	#sink_s = gr.vector_sink_c(M)

	self.tb.connect(src, self.cs_f, sink_f)
	#self.tb.connect(src, self.cs_s, sink_s)

	self.tb.run()

	output_data_f = sink_f.data()

	self.assertEqual(output_data_f, (1, 8.5, 1, 8.5))
	#self.assertEqual(output_data_s, (1, 8, 1, 8))


    def test_002_random (self):
	M = 16
	N = 64
	B = 5
	iter = 5


	for k in range(iter):
		input_data = self.random_complex_data(B * N)
		matrix = random.randn(M, N)

		src = gr.vector_source_c(input_data, False, N)
		sink = gr.vector_sink_c(M)
		cs_gen = cs.generic_vccf(matrix)

		self.tb.connect(src, cs_gen, sink)
		self.tb.run()

		output_data = sink.data()

		correct_output = list()
		for i in range(B):
			correct_output.extend(dot(matrix, input_data[i*N:(i+1)*N]))

		self.assertComplexTuplesAlmostEqual(output_data, correct_output, 4)

		self.tb.disconnect_all()



    def random_complex_data (self, n):
	data = list()
	for i in range(n):
		data.append(complex(random.randn(), random.randn()))

	return data



if __name__ == '__main__':
	gr_unittest.main ()
