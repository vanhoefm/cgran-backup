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

import numpy
from gnuradio import gr, gr_unittest
import itpp_swig as itpp

class qa_egolay (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_encode (self):
        src_data = (0, ) * 12
        expected_result = (0,) * 24

        src = gr.vector_source_b (src_data, False, 12)
        coder = itpp.egolay_encoder_vbb()
        dst = gr.vector_sink_b (24)
        self.tb.connect (src, coder, dst)
        self.tb.run ()
        result_data = dst.data ()
        self.assertEqual (expected_result, result_data)

    def test_004_encdec (self):
        """ Stream n_words through the extended Golay encoder, corrupt two bits per code word,
        decode, and compare the result with the initial sequence. """
        n = 24
        k = 12
        n_words = 1000
        src_data = tuple([int(x) for x in numpy.random.randint(0, 2, n_words * k)])
        error_data = (1,1,) + (0,) * (n-2)

        src = gr.vector_source_b (src_data, False, k)
        encoder = itpp.egolay_encoder_vbb()
        v2s = gr.vector_to_stream(gr.sizeof_char, n)
        noise_src = gr.vector_source_b (error_data, True)
        chan = gr.xor_bb()
        s2v = gr.stream_to_vector(gr.sizeof_char, n)
        decoder = itpp.egolay_decoder_vbb()
        dst = gr.vector_sink_b (k)
        self.tb.connect (src, encoder, v2s, chan, s2v, decoder, dst)
        self.tb.connect(noise_src, (chan, 1))

        self.tb.run ()
        result_data = dst.data ()
        self.assertEqual (src_data, result_data)
        
if __name__ == '__main__':
    gr_unittest.main ()
