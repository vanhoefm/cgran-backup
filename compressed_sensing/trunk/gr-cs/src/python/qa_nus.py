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


class qa_nus(gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()
        self.compression = 4
        self.nus = cs.nusaic_cc(self.compression)

    def tearDown (self):
        self.tb = None
        self.nus = None

    def test_001_deterministic (self):
        input_data = (0+0j, 1+0j, 2+0j, 3+0j, 0+0j, 1+0j, 2+0j, 3+0j)
        select_data = (1, 3)
        sample_src = gr.vector_source_c(input_data, False)
        offset_src = gr.vector_source_i(select_data, False)

        output = gr.vector_sink_c()

        self.tb.connect(sample_src, (self.nus, 0))
        self.tb.connect(offset_src, (self.nus, 1))
        self.tb.connect(self.nus, output)

        self.tb.run()

        output_data = output.data()

        self.assertEqual(len(input_data) / self.compression, len(output_data))
        metric = 0
        for i in range(len(select_data)):
            print i
            metric += select_data[i-1] - output_data[i-1]

        self.assertEqual(metric, 0)



if __name__ == '__main__':
    gr_unittest.main ()
