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
import threading
import time
import sys

if(len(sys.argv)>1):
    TOTAL_LENGTH = int(sys.argv[1])
else:
    TOTAL_LENGTH = 100000

class CPU_Thread(threading.Thread):
    """CPU Thread"""
    def __init__(self):
        threading.Thread.__init__(self)
        self.tb = gr.top_block ()
        src_data = list(math.sin(x) for x in range(TOTAL_LENGTH))
        src = gr.vector_source_f (src_data)
        taps = [0]*60
        taps[0]=.5
        taps[1]=.5
        op  = gr.fir_filter_fff(1,taps)
        op2  = gr.fir_filter_fff(1,taps)
        self.dst = gr.vector_sink_f ()
        self.tb.connect(src, op)
        self.tb.connect(op,  op2)
        self.tb.connect(op2,  self.dst)

    def run(self):
        print "=============== runtime log ============="
        self.tb.run ()        
        print "========================================="
        cpuonly_data = self.dst.data()
        print "cpu out:",list(cpuonly_data[i] for i in range(5))
        
class GPU_Thread(threading.Thread):
    """GPU Thread"""
    def __init__(self):
        threading.Thread.__init__(self)
        self.tb = gr.top_block ()
        src_data = list(math.sin(x) for x in range(TOTAL_LENGTH))
        src = gr.vector_source_f (src_data)
        h2d = grgpu.h2d_cuda()
        taps = [0]*60
        taps[0]=.5
        taps[1]=.5
        op  = grgpu.fir_filter_fff_cuda(taps)
        #op.set_verbose(1)
        op2  = grgpu.fir_filter_fff_cuda(taps)
        d2h = grgpu.d2h_cuda()
        self.dst = gr.vector_sink_f ()
        self.tb.connect(src, h2d)
        self.tb.connect(h2d, op)
        self.tb.connect(op,  op2)
        self.tb.connect(op2,  d2h)
        self.tb.connect(d2h, self.dst)

    def run(self):
        print "=============== runtime log ============="
        self.tb.run ()        
        print "========================================="
        result_data = self.dst.data ()
        
        print "gpu out:",list(result_data[i] for i in range(5))


#        expected_result =         cpuonly_data[1:10000]
#result_data =         result_data[0:9999]

src_data = list(math.sin(x) for x in range(10000))
#print "================= results ==============="
print "in:",list(src_data[i] for i in range(5))
#print "========================================="

#self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)
cpu_thread = CPU_Thread()
gpu_thread = GPU_Thread()

start = time.time()
cpu_thread.start()
cpu_thread.join()
print "CPU Elapsed Time: %s" % (time.time() - start)
gpu_thread.start()
gpu_thread.join()
print "GPU Elapsed Time: %s" % (time.time() - start)
