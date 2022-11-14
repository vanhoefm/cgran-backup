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

class qa_bessel (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_besselj (self):
        src_data = (1, 2, 3, 4)
        expected_data = (0.7652, 0.2239, -0.2601, -0.3971)

        src = gr.vector_source_f(src_data)
        bessel = itpp.besselj_ff()
        dst = gr.vector_sink_f()

        self.tb.connect(src, bessel, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(result_data, expected_data, 4)


    def test_002_besseli (self):
        src_data = (1, 2, 3, 4)
        expected_data = (1.2661, 2.2796, 4.8808,11.3019)

        src = gr.vector_source_f(src_data)
        bessel = itpp.besseli_ff()
        dst = gr.vector_sink_f()

        self.tb.connect(src, bessel, dst)
        self.tb.run()
        result_data = dst.data()
        self.assertFloatTuplesAlmostEqual(result_data, expected_data, 4)


        
if __name__ == '__main__':
    gr_unittest.main ()
