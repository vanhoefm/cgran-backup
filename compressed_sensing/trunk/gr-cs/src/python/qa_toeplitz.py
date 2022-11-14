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

class qa_cs_toeplitz(gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()


    def tearDown (self):
        self.tb = None
        self.cs = None

    def test_001_deterministic (self):
        sequence = (0, 1, 1,    0,    1,    1,    0,   0, 1, 1, 1, 0)
        input_data =     (1+0j, 0+0j, 3+0j, 2+0j, 1+0j) * 2

        M = 3
        N = 5

        self.cs = cs.toeplitz_vccb(N, M, sequence[0:M-1])

        src_data = gr.vector_source_c(input_data, False, N)
        src_seq = gr.vector_source_b(sequence[M-1:], False, N)
        sink = gr.vector_sink_c(M)

        self.tb.connect(src_data, (self.cs, 0), sink)
        self.tb.connect(src_seq, (self.cs, 1))
        self.tb.run()

        output_data = sink.data()

        self.assertEqual(output_data, (5, 1, 1, 3, 5, 1))


    def test_002_defaultconstructor (self):
        sequence = (0, 1, 1,    0,    1,    1,    0,   0, 1, 1, 1, 0)
        input_data =     (1+0j, 0+0j, 3+0j, 2+0j, 1+0j) * 2

        M = 3
        N = 5

        self.cs = cs.toeplitz_vccb(N, M, list())

        src_data = gr.vector_source_c(input_data, False, N)
        src_seq = gr.vector_source_b(sequence[M-1:], False, N)
        sink = gr.vector_sink_c(M)

        self.tb.connect(src_data, (self.cs, 0), sink)
        self.tb.connect(src_seq, (self.cs, 1))
        self.tb.run()

        output_data = sink.data()

        self.assertEqual(output_data, (5, -1, 1, 3, 5, 1))


    def test_003_checkinput (self):
        self.assertRaises(ValueError, cs.toeplitz_vccb, 5, 3, (1,))


    def test_004_random (self):
        B = 3
        for M in range(5, 20, 2):
            for N in range(M+1, M+10, 3):
                sequence = self.random_sequence(B * N + M-1)
                data = self.random_complex_data(B * N)
                correct_output = self.calc_toeplitz_output(sequence, data, M, N)

                src_data = gr.vector_source_c(data, False, N)
                src_seq = gr.vector_source_b(sequence[M-1:], False, N)
                sink = gr.vector_sink_c(M)
                self.cs = cs.toeplitz_vccb(N, M, sequence[0:M-1])

                self.tb.connect(src_data, (self.cs, 0), sink)
                self.tb.connect(src_seq, (self.cs, 1))
                self.tb.run()

                output_data = sink.data()
                self.assertComplexTuplesAlmostEqual(output_data, tuple(correct_output), 5)

                self.tb.disconnect_all()


    def calc_toeplitz_output (self, in_seq, data, M, N):
        result = list()
        sequence = list()
        pm = (-1, 1)
        for i in range(len(in_seq)):
            sequence.append(pm[in_seq[i]])

        n_blocks = len(data) / N
        for i in range(n_blocks):
            for k in range(M):
                result.append(dot(data[N*i:N*(i+1)], sequence[N*i+M-1-k:N*(i+1)+M-1-k]))

        return result


    def random_sequence (self, n):
        seq = list()
        seq = random.randint(0, 2, n)

        return seq


    def random_complex_data (self, n):
        data = list()
        for i in range(n):
            data.append(complex(random.randn(), random.randn()))

        return data


if __name__ == '__main__':
    gr_unittest.main ()
