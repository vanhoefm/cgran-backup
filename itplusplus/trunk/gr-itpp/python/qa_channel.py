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

class qa_channel (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_channelenergy (self):
        """ Check if the energy of a received signal is lower than the transmitted signal. """
        vlen = 128
        vpad = 128
        src = gr.noise_source_c(gr.GR_GAUSSIAN, 1)
        head = gr.head(gr.sizeof_gr_complex, 1024)
        s2v = gr.stream_to_vector(gr.sizeof_gr_complex, vlen)
        chan = itpp.channel_tdl_vcc(vlen, vpad, itpp.COST207_BU, 1e-9)
        magsqr = gr.complex_to_mag_squared(vlen + vpad)
        dst = gr.vector_sink_f(vlen + vpad)

        self.tb.connect(src, head, s2v, chan, magsqr, dst)
        self.tb.run()

        result_data = dst.data()

        power = numpy.sum(result_data) / len(result_data)
        self.assertTrue(power < 1)

        
if __name__ == '__main__':
    gr_unittest.main ()
