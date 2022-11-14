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

class qa_cs_circmat(gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()


    def tearDown (self):
        self.tb = None
        self.cs = None


    def test_001_deterministic (self):
        input_data = (0+0j, 1+0j, 2+0j, 0+0j, 1+0j, 2+0j)

        M = 2
        N = 3
        sequence = (-1, 0, 1)

        self.cs = cs.circmat_vccb(sequence, M)
        src = gr.vector_source_c(input_data, False, N)
        sink = gr.vector_sink_c(M)

        self.tb.connect(src, self.cs, sink)
        self.tb.run()

        output_data = sink.data()
        self.assertEqual(output_data, (2, -1, 2, -1))


    def test_002_translate (self):
        input_data = (0+0j, 1+0j, 2+0j, 0+0j, 1+0j, 2+0j)

        M = 2
        N = 3
        sequence = (0, 0, 1)

        self.cs = cs.circmat_vccb(sequence, M, True)
        src = gr.vector_source_c(input_data, False, N)
        sink = gr.vector_sink_c(M)

        self.tb.connect(src, self.cs, sink)
        self.tb.run()

        output_data = sink.data()
        self.assertEqual(output_data, (1, -3, 1, -3))


    def test_003_random (self):
        M = 16
        N = 64
        B = 5
        iter = 5

        for k in range(iter):
            input_data = self.random_complex_data(B * N)
            sequence = list(random.randint(0, 2, N))
            matrix = self.matrix_from_sequence(sequence, N, M, True)

            src = gr.vector_source_c(input_data, False, N)
            sink = gr.vector_sink_c(M)
            cs_circ = cs.circmat_vccb(sequence, M, True)

            self.tb.connect(src, cs_circ, sink)
            self.tb.run()

            output_data = sink.data()

            correct_output = list()
            for i in range(B):
                correct_output.extend(dot(matrix, input_data[i*N:(i+1)*N]))

            self.assertComplexTuplesAlmostEqual(output_data, correct_output, 4)

            self.tb.disconnect_all()


    def matrix_from_sequence (self, sequence, N, M, translate):
        matrix = zeros((M, N), dtype=int)

        for i in range(M):
            for j in range(N):
                elmt = sequence[(N+j-i) % N]
                if translate and elmt == 0:
                    elmt = -1
                matrix[i][j] = elmt

        return matrix


    def random_complex_data (self, n):
        data = list()
        for i in range(n):
            data.append(complex(random.randn(), random.randn()))

        return data



if __name__ == '__main__':
    gr_unittest.main ()
