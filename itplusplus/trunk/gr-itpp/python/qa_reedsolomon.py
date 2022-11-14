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

from gnuradio import gr, gr_unittest
import itpp_swig as itpp

class qa_reedsolomon (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_encode (self):
        m = 3 # Results in (7, 3) Reed-Solomon Code
        t = 2
        n = 7
        k = 3
        src_data = (0, ) * (k * m)
        expected_result = (0,) * (n * m)

        src = gr.vector_source_b (src_data, False, k * m)
        coder = itpp.reedsolomon_encoder_vbb(m, t)
        self.assertEqual(coder.get_n(), n)
        self.assertEqual(coder.get_k(), k)
        dst = gr.vector_sink_b (n * m)
        self.tb.connect (src, coder, dst)
        self.tb.run ()
        result_data = dst.data ()
        self.assertEqual (expected_result, result_data)

    def test_002_exception (self):
        self.assertRaises(ValueError, lambda: itpp.reedsolomon_encoder_vbb(0, 1))

    
    def test_003_encdec (self):
        pass

        
if __name__ == '__main__':
    gr_unittest.main ()
