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

from gnuradio import gr
import grgpu
import math


tb = gr.top_block ()
src_data = list(math.sin(x) for x in range(10000))
src = gr.vector_source_f (src_data)
src2 = gr.vector_source_f (src_data)
h2d = grgpu.h2d_cuda()
h2d2 = grgpu.h2d_cuda()
op  = grgpu.add_const_ff_cuda()
op2  = grgpu.add_const_ff_cuda()
d2h = grgpu.d2h_cuda()
d2h2 = grgpu.d2h_cuda()
dst = gr.vector_sink_f ()
dst2 = gr.vector_sink_f ()
tb.connect (src, h2d)
tb.connect (h2d, op)
tb.connect (op,  d2h)
tb.connect (d2h, dst)

tb.connect (src2, h2d2)
tb.connect (h2d2, op2)
tb.connect (op2,  d2h2)
tb.connect (d2h2, dst2)
print "=============== runtime log ============="
tb.run ()        
print "========================================="
result_data = dst.data ()

print "================= results ==============="
print "in:",list(src_data[i] for i in range(5))
print "out:",list(result_data[i] for i in range(5))
print "========================================="


