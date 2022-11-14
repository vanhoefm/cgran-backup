#!/usr/bin/env python
#
# Copyright 2004,2007 Free Software Foundation, Inc.
# 
# This file is part of GNU Radio
# 
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
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

import numpy.random
from gnuradio import gr, gr_unittest
import itpp_swig as itpp

class qa_hamming (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_encode (self):
        m = 3 # Results in (7, 4) Hamming code
        n = 7
        k = 4
        src_data = (0, ) * k
        expected_result = (0,) * n

        src = gr.vector_source_b (src_data, False, k)
        coder = itpp.hamming_encoder_vbb(m)
        dst = gr.vector_sink_b (n)
        self.tb.connect (src, coder, dst)
        self.tb.run ()
        result_data = dst.data ()
        self.assertEqual (expected_result, result_data)

    def test_002_decode (self):
        m = 3 # Results in (7, 4) Hamming code
        n = 7
        k = 4
        src_data = (1,) + (0, ) * (n-1)
        expected_result = (0,) * k

        src = gr.vector_source_b (src_data, False, n)
        coder = itpp.hamming_decoder_vbb(m)
        dst = gr.vector_sink_b (k)
        self.tb.connect (src, coder, dst)
        self.tb.run ()
        result_data = dst.data ()
        self.assertEqual (expected_result, result_data)

    def test_003_exception (self):
        self.assertRaises(ValueError, lambda: itpp.hamming_encoder_vbb(0))

    def test_004_encdec (self):
        """ Stream n_words through a given hamming encoder, corrupt one bit per code word,
        decode, and compare the result with the initial sequence. """
        m = 3 # Results in (7, 4) Hamming code
        n = 7
        k = 4
        n_words = 100
        src_data = tuple([int(x) for x in numpy.random.randint(0, 2, n_words * k)])
        error_data = (1,) + (0,) * (n-1)

        src = gr.vector_source_b (src_data, False, k)
        encoder = itpp.hamming_encoder_vbb(m)
        v2s = gr.vector_to_stream(gr.sizeof_char, n)
        noise_src = gr.vector_source_b (error_data, True)
        chan = gr.xor_bb()
        s2v = gr.stream_to_vector(gr.sizeof_char, n)
        decoder = itpp.hamming_decoder_vbb(m)
        dst = gr.vector_sink_b (k)
        self.tb.connect (src, encoder, v2s, chan, s2v, decoder, dst)
        self.tb.connect(noise_src, (chan, 1))

        self.tb.run ()
        result_data = dst.data ()
        self.assertEqual (src_data, result_data)

        
if __name__ == '__main__':
    gr_unittest.main ()
