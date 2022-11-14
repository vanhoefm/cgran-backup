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
#from grc_gnuradio import wxgui as grc_wxgui
#from gnuradio.wxgui import fftsink2
import grgpu
import math
from pylab import *
import wx

tb = gr.top_block ()
#+ .5*math.sin(x/600.0) + .5* math.sin(x/100.0)
#src_data = list((math.sin(x/1000.0) + .5*math.sin(x/600.0) + .5* math.sin(x/100.0)) for x in range(1024*8))
src_data = list((math.sin(x/1000.0) + math.sin(x/400.0) + math.sin(x/300.0)  ) for x in range(1024*32))
#src_data = [0.0]*(100)
#src_data = list((math.sin(x/1000.0) + .5*math.sin(x/600.0) + .5* math.sin(x/100.0)) for x in range(1024*320))
src = gr.vector_source_f (src_data)
h2d = grgpu.h2d_cuda()
h2d.set_verbose(1)
#op  = grgpu.fft_vfc_cuda()
#op  = grgpu.add_const_ff_cuda(.4)
#op  = grgpu.resampler_fff_cuda()
d2h = grgpu.d2h_cuda()
d2h.set_verbose(1)

dst = gr.vector_sink_f ()

tb.connect (src, h2d)
tb.connect (h2d, d2h)
#tb.connect (h2d, op, d2h)
tb.connect (d2h, dst)
print "=============== runtime log ============="
tb.run ()        
print "========================================="
result_data = dst.data ()
result_list = range(20000)
src_list = range(20000)
#switch the tuple over to a list and clip the result data for display purposes
for i in range(20000):
    result_list[i] = result_data[i]
    src_list[i] = src_data[i]
    if result_list[i]>5:
        result_list[i]=5
    if result_list[i]<-5:
        result_list[i]=-5


print "================= results ==============="
print "in:",list(src_data[i] for i in range(10))
print "out:",list(result_data[i] for i in range(10))
print "========================================="

x = range(20000)
plot(x, src_list, 'r-')
grid(True)

x = range(20000)
plot(x, result_list, 'b-')
show()
