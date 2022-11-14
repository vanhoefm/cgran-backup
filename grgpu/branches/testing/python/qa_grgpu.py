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
import grgpu
import math

class qa_grgpu (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()

    def tearDown (self):
        self.tb = None

    def test_001_fir_fff_cuda (self):
        src_data = list(math.sin(x) for x in range(10240))

#run the cuda version first to catch errors faster 
#also, this test uses old version of grgpu actor (stand alone actors)
        src = gr.vector_source_f (src_data)
        op = grgpu.fir_fff_cuda ()
        dst = gr.vector_sink_f ()
        self.tb.connect (src, op)
        self.tb.connect (op, dst)
        self.tb.run ()        
        result_data = dst.data ()

        print "Now running reference"
        # run the reference
        self.tb = gr.top_block ()
        src = gr.vector_source_f (src_data)
        dst = gr.vector_sink_f ()
        taps = 60*[0]
        taps[0] = 0.5
        taps[1] = 0.5
        op = gr.fir_filter_fff(1, taps)
        self.tb.connect (src, op)
        self.tb.connect (op, dst)
        self.tb.run ()        
        expected_result = dst.data ()

        expected_result =         expected_result[1:10000]
        result_data =         result_data[0:9999]

        self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)

    def test_002_h2d_d2h_cuda (self):
        print "running test2"
        src_data = list(math.sin(x) for x in range(10240))

#run the cuda version first to catch errors faster
        src = gr.vector_source_f (src_data)
        op = grgpu.h2d_cuda()
        op2 = grgpu.d2h_cuda()
        dst = gr.vector_sink_f ()
        self.tb.connect (src, op)
        self.tb.connect (op, op2)
        self.tb.connect (op2, dst)
        self.tb.run ()        
        result_data = dst.data ()
        
        print "in002:",list(src_data[i] for i in range(5))
        print "out002:",list(result_data[i] for i in range(5))
        self.assertFloatTuplesAlmostEqual (src_data, result_data, 6)

    def test_003_add_const_ff_cuda (self):
        print "running test3"
        #ref
        src_data = list(math.sin(x) for x in range(10240))
        src = gr.vector_source_f (src_data)
        op  = gr.add_const_ff(.2)
        dst = gr.vector_sink_f ()
        self.tb.connect (src, op)
        self.tb.connect (op, dst)
        self.tb.run ()        
        ref_data = dst.data ()
        

        #cuda
        src_data = list(math.sin(x) for x in range(10240))
        src = gr.vector_source_f (src_data)
        h2d = grgpu.h2d_cuda()
        op  = grgpu.add_const_ff_cuda(.2)
        d2h = grgpu.d2h_cuda()
        dst = gr.vector_sink_f ()
        self.tb.connect (src, h2d)
        self.tb.connect (h2d, op)
        self.tb.connect (op,  d2h)
        self.tb.connect (d2h, dst)
        self.tb.run ()        
        result_data = dst.data ()

        print "in003:",list(src_data[i] for i in range(5))
        print "ref003:",list(ref_data[i] for i in range(5))
        print "out003:",list(result_data[i] for i in range(5))
        self.assertFloatTuplesAlmostEqual (ref_data, result_data, 6)


    def test_004_2_add_const_ff_cuda (self):
        #ref
        src_data = list(math.sin(x) for x in range(10240))
        src = gr.vector_source_f (src_data)
        op  = gr.add_const_ff(.1)
        op2  = gr.add_const_ff(.2)
        dst = gr.vector_sink_f ()
        self.tb.connect (src, op)
        self.tb.connect (op, op2)
        self.tb.connect (op2, dst)
        self.tb.run ()        
        ref_data = dst.data ()
        

        #cuda
        src_data = list(math.sin(x) for x in range(10240))
        src = gr.vector_source_f (src_data)
        h2d = grgpu.h2d_cuda()
        op  = grgpu.add_const_ff_cuda(.1)
        op2  = grgpu.add_const_ff_cuda(.2)
        d2h = grgpu.d2h_cuda()
        dst = gr.vector_sink_f ()
        self.tb.connect (src, h2d)
        self.tb.connect (h2d, op)
        self.tb.connect (op, op2)
        self.tb.connect (op2,  d2h)
        self.tb.connect (d2h, dst)
        self.tb.run ()        
        result_data = dst.data ()

        print "in004:",list(src_data[i] for i in range(5))
        print "ref004:",list(ref_data[i] for i in range(5))
        print "out004:",list(result_data[i] for i in range(5))
        self.assertFloatTuplesAlmostEqual (ref_data, result_data, 6)


 #     def test_005_fir_filter_fff_cuda (self):
 #         src_data = list(math.sin(x) for x in range(10240))

 # #run the cuda version first to catch errors faster
 #         src = gr.vector_source_f (src_data)
 #         op = grgpu.fir_filter_fff_cuda ()
 #         dst = gr.vector_sink_f ()
 #         self.tb.connect (src, op)
 #         self.tb.connect (op, dst)
 #         self.tb.run ()        
 #         result_data = dst.data ()

 #         print "Now running reference"
 #         # run the reference
 #         self.tb = gr.top_block ()
 #         taps = 60*[0]
 #         taps[0] = 1.0
 #         fir = gr.fir_filter_fff(1, taps)
 #         self.tb.connect (src, op)
 #         self.tb.connect (op, dst)
 #         self.tb.run ()        
 #         expected_result = dst.data ()
       
 #         print list(expected_result[i] for i in range(10))
 #         print list(result_data[i] for i in range(10))
 #         self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)

        
if __name__ == '__main__':
    gr_unittest.main ()
