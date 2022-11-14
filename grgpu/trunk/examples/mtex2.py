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

from gnuradio import gr, window
import grgpu
import math
from multiprocessing import Process, Pipe
import time
import sys
import os
import struct

#threads are paired for now
num_threads = 2
num_gpu_threads = 1
#do break, return, and stop
splice = [0]*(10)
#in each case we take 2 stages to either vectorize or h2d/d2h the data
stages = 2+2
splice[0] = [1, stages,stages+1]
splice[1] = [1, stages,stages+1]
splice[2] = [1,stages,stages+1]
splice[3] = [1,stages,stages+1]

if(len(sys.argv)>1):
    TOTAL_LENGTH = int(sys.argv[1])
else:
    TOTAL_LENGTH = 99328 #100000

CHUNK_LENGTH = 32768
if TOTAL_LENGTH>CHUNK_LENGTH:
    NUM_CHUNKS = int(TOTAL_LENGTH/CHUNK_LENGTH)
else:
    CHUNK_LENGTH = TOTAL_LENGTH
    NUM_CHUNKS = 1


class all_CPU_Thread(Process):
    """All CPU Thread"""
    def __init__(self, tid, stop):
        Process.__init__(self)
        self.stop = stop
        self.tid = tid
        self.tb = gr.top_block ()
        self.op = [0]*self.stop

        src_data = list(math.sin(x) for x in range(CHUNK_LENGTH))
#        self.src = gr.file_descriptor_source( 4,self.inputpipe.fileno(), False)
        self.src = gr.vector_source_f(src_data*NUM_CHUNKS)
        
        for j in range(self.stop):
            self.op[j]  = gr.add_const_ff(.4)

        fftlen = CHUNK_LENGTH
        mywindow = window.blackmanharris(fftlen)
        self.op[0] = gr.stream_to_vector(4, fftlen)
        self.op[1] = gr.fft_vfc(fftlen, True, mywindow )
        for i in range(2,self.stop-1):
            self.op[i] = gr.fft_vcc(fftlen, True, mywindow )
        self.op[self.stop-1] = gr.vector_to_stream(8, fftlen)   
        self.dst = gr.vector_sink_c ()

#        taps = [.02]*80
#        for i in range(self.stop):
            #self.op[i] = gr.add_const_ff(.01)
#            self.op[i] = gr.fir_filter_fff(1,taps)
#        self.dst = gr.vector_sink_f ()

        self.tb.connect(self.src,self.op[0])

        for j in range(self.stop-1):
            self.tb.connect(self.op[j],self.op[j+1])
            

        self.tb.connect(self.op[self.stop-1],  self.dst)

    def run(self):
        print "=============== CPU Thread %d ============="%self.tid
 
        start = time.time()
        self.tb.start()
        self.tb.wait()
        cpu_time = time.time() - start 

        print "=================CPU Thread %d Done =============="%self.tid
        self.data = self.dst.data()
#        print "%d cpu out:"%self.tid,list(self.data[i] for i in range(5))
        print "all CPU internal Elapsed Time: %s" % (cpu_time)


class all_GPU_Thread(Process):
    """All GPU Thread"""
    def __init__(self, tid, stop):
        Process.__init__(self)
        self.stop = stop
        self.tid = tid
        self.tb = gr.top_block ()
        self.op = [0]*self.stop

        src_data = list(math.sin(x) for x in range(CHUNK_LENGTH))
#        self.src = gr.file_descriptor_source( 4,self.inputpipe.fileno(), False)
        self.src = gr.vector_source_f(src_data*NUM_CHUNKS)
        
        fftlen = CHUNK_LENGTH
        mywindow = window.blackmanharris(fftlen)
        self.op[0] = grgpu.h2d_cuda()
        self.op[0].set_verbose(1)
        for i in range(1,self.stop-1):
#            self.op[i] = grgpu.add_const_ff_cuda(.01)
            self.op[i] = grgpu.fft_vfc_cuda()
        self.op[self.stop-1] = grgpu.d2h_cuda()
        self.tb.connect(self.src,self.op[0])

        for j in range(self.stop-1):
            self.tb.connect(self.op[j],self.op[j+1])
            
        self.dst = gr.vector_sink_f ()
        self.tb.connect(self.op[self.stop-1],  self.dst)

    def run(self):
        print "=============== GPU Thread %d ============="%self.tid
 
        start = time.time()
        self.tb.start()
        self.tb.wait()
        cpu_time = time.time() - start 

        print "================= GPU Thread %d Done =============="%self.tid
        self.data = self.dst.data()
#        print "%d cpu out:"%self.tid,list(self.data[i] for i in range(5))
        print "all GPU internal Elapsed Time: %s" % (cpu_time)


class CPU_Thread(Process):
    """CPU Thread"""
    def __init__(self, tid, splice, pipe_to_gpu, pipe_from_gpu, killpipe, datapipe, inputpipe):
        Process.__init__(self)
        self.tid = tid
        self.datapipe = datapipe
        self.inputpipe = inputpipe
        self.killpipe = killpipe
        self.pipe_to_gpu = pipe_to_gpu
        self.pipe_from_gpu = pipe_from_gpu
        self.tb = gr.top_block ()
        self.splice = splice
        self.gpu_start = splice[0]
        self.restart = splice[1]
        self.stop = splice[2]
        self.op = [0]*self.stop

        self.src = gr.file_descriptor_source( 4,self.inputpipe.fileno(), False)
        for j in range(self.gpu_start):
            self.op[j]  = gr.add_const_ff(.4)
            
        self.tb.connect(self.src,self.op[0])
        for j in range(1,self.gpu_start):
            self.tb.connect(self.op[j-1],self.op[j])

        for j in range(self.restart,self.stop):
            self.op[j]  = gr.add_const_ff(.4)
        #if there is a non trivial splice
        if self.gpu_start!=self.restart:
            #send over to the gpu thread
            self.togpu   = gr.file_descriptor_sink(4,self.pipe_to_gpu.fileno())
            self.tb.connect(self.op[self.gpu_start-1],self.togpu)
            #get back from the gpu thread
            self.fromgpu = gr.file_descriptor_source(4, self.pipe_from_gpu.fileno(),False)
            self.tb.connect(self.fromgpu, self.op[self.restart])
        else:
            self.tb.connect(self.op[self.gpu_start-1],self.op[self.restart])

        for j in range(self.restart,self.stop-1):
            self.tb.connect(self.op[j],self.op[j+1])
            
        self.dst = gr.vector_sink_f ()
        self.tb.connect(self.op[self.stop-1],  self.dst)

    def run(self):
        print "=============== CPU Thread %d ============="%self.tid
 
        start = time.time()
        self.tb.start()
        ll = -1
#each cpu thread is to kill its GPU pair thread
        while len(self.dst.data())<1:
            if(ll!=len(self.dst.data())):
                ll = len(self.dst.data())
                print ll
            time.sleep(.0001)
        sofar = len(self.dst.data())
        print sofar, time.time()-start
        start = time.time()
#each cpu thread is to kill its GPU pair thread
        while len(self.dst.data())<(TOTAL_LENGTH):
            if(ll!=len(self.dst.data())):
                ll = len(self.dst.data())
                print ll
            time.sleep(.001)

        cpu_time = time.time() - start 
        
        self.killpipe.send_bytes('d')
        self.tb.stop()


#        while len(self.dst.data())<TOTAL_LENGTH:
#            print len(self.dst.data())
#            time.sleep(.01)
        print "=================CPU Thread %d Done =============="%self.tid
        self.data = self.dst.data()
        self.datapipe.send(self.data)
        print "%d cpu out:"%self.tid,list(self.data[i] for i in range(5))
        print "CPU Elapsed Time: %s" % (cpu_time)
       
class CPU_Thread2(Process):
    """CPU Thread Emulating GPU functionality Thread"""
    def __init__(self, tid, splice, to_gpu_pipe, from_gpu_pipe, killpipe):
        Process.__init__(self)
        self.tid = tid
        self.tb = gr.top_block ()
        self.fdin = to_gpu_pipe
        self.fdout = from_gpu_pipe
        self.killpipe = killpipe
        self.splice = splice
        self.gpu_start = splice[0]
        self.restart = splice[1]
        self.stop = splice[2]

        src = gr.file_descriptor_source( 4,self.fdin.fileno(), False)
        dst = gr.file_descriptor_sink(8, self.fdout.fileno())
        fftlen = CHUNK_LENGTH

        length = self.restart - self.gpu_start

        op = [0]*length
        mywindow = window.blackmanharris(fftlen)
        op[0] = gr.stream_to_vector(4, fftlen)
        op[1] = gr.fft_vfc(fftlen, True, mywindow )
        for i in range(2,length-1):
            op[i] = gr.fft_vcc(fftlen, True, mywindow )
        op[length-1] = gr.vector_to_stream(8, fftlen)   

        if length>0:
            self.tb.connect(src, op[0])
            for i in range(length-1):
                self.tb.connect(op[i],op[i+1])
            self.tb.connect(op[length-1],dst)
        else:
            self.tb.connect(src,dst)

    def run(self):
        print "=============CPU2 Thread %d log================"%self.tid
        self.tb.start()

            #wait for the end task to kill me
        self.killpipe.poll(None)
        print "CPU2 Thread %d got killed!"%self.tid

        self.tb.stop()
        print "===============CPU2 Thread %d Done=================="%self.tid
        
        
        print "Thread %d:"%self.tid
        
        
class GPU_Thread(Process):
    """GPU Thread"""
    def __init__(self, tid, splice, to_gpu_pipe, from_gpu_pipe, killpipe):
        Process.__init__(self)
        self.tid = tid
        self.tb = gr.top_block ()
        self.fdin = to_gpu_pipe
        self.fdout = from_gpu_pipe
        self.killpipe = killpipe
        self.splice = splice
        self.gpu_start = splice[0]
        self.restart = splice[1]
        self.stop = splice[2]

        os.environ['GR_SCHEDULER']='STS'

        src = gr.file_descriptor_source( 4,self.fdin.fileno(), False)
        h2d = grgpu.h2d_cuda()
        d2h = grgpu.d2h_cuda()
        dst = gr.file_descriptor_sink(4, self.fdout.fileno())

        length = self.restart-self.gpu_start

        op = [0]*length
        for i in range(length):
            op[i] = grgpu.fft_vfc_cuda()
#            op[i]  = grgpu.add_const_ff_cuda(.4)

        self.tb.connect(src, h2d)
        if length>0:
            self.tb.connect(h2d, op[0])
            for i in range(length-1):
                self.tb.connect(op[i],op[i+1])
            self.tb.connect(op[-1],d2h)
        else:
            self.tb.connect(h2d,d2h)
        self.tb.connect(d2h, dst)

    def run(self):
        os.environ['GR_SCHEDULER']='STS'
        print "=============GPU Thread %d log================"%self.tid
        self.tb.start()

            #wait for the end task to kill me
        self.killpipe.poll(None)
        print "GPU Thread %d got killed!"%self.tid

        self.tb.stop()
        print "===============GPU Thread %d Done=================="%self.tid
        
        
        print "Thread %d:"%self.tid

src_data = list(math.sin(x) for x in range(1000))
#print "================= results ==============="
print "in:",list(src_data[i] for i in range(5))
#print "========================================="


#self.assertFloatTuplesAlmostEqual (expected_result, result_data, 6)
cpudatapipeout, cpudatapipein = Pipe()
cpu_threads = [0]*(num_threads-1)
gpu_threads = [0]*(num_threads-1)
cpu1_threads = [0]
cpu2_threads = [0]
cpudata = [0]*num_threads
fdtogpuin  = [0]*num_threads
fdtogpuout = [0]*num_threads
fdfromgpuin = [0]*num_threads
fdfromgpuout= [0]*num_threads
killout   = [0]*num_threads
killin    = [0]*num_threads
datapipeout = [0]*num_threads
datapipein = [0]*num_threads
inputpipein = [0]*num_threads
inputpipeout = [0]*num_threads
for i in range(num_threads):
    #return results pipe
    datapipeout[i], datapipein[i] = Pipe()
    inputpipeout[i], inputpipein[i] = Pipe()
    #the comm pipe
    fdtogpuout[i], fdtogpuin[i] = Pipe()
    fdfromgpuout[i], fdfromgpuin[i] = Pipe()
    #the kill pipe
    killout[i], killin[i] = Pipe()
for i in range(num_gpu_threads):
    cpu_threads[i] = CPU_Thread(i, splice[i], fdtogpuout[i],fdfromgpuin[i],killout[i],datapipeout[i], inputpipeout[i])
#    gpu_threads[i] = GPU_Thread(i, splice[i], fdtogpuin[i],fdfromgpuout[i],killin[i])

i = num_threads-1
cpu1_threads[0] = CPU_Thread(i, splice[i], fdtogpuout[i],fdfromgpuin[i],killout[i],datapipeout[i], inputpipeout[i])
cpu2_threads[0] = CPU_Thread2(i, splice[i], fdtogpuin[i],fdfromgpuout[i],killin[i])


#src_data = list(complex(math.sin(x/1000.0) + .5*math.sin(x/600.0) + .5* math.sin(x/100.0)) for x in range(1024*20))

prep_data = apply(struct.pack,
                 ["f"*32]+ list(math.sin(x) for x in range(32)))

src_data = apply(struct.pack,
                 ["f"*CHUNK_LENGTH]+ list(math.sin(x) for x in range(CHUNK_LENGTH)))

print len(src_data)

# ### execute section ###
# #CPU
# start = time.time()
# for ct,gt in zip(cpu_threads,gpu_threads):
#     ct.start()
#     gt.start()

# #for i in range(num_threads):
# #    print "warming up the pipeline..."
# #    inputpipein[i].send_bytes(prep_data)

# time.sleep(1.3)
# print "done sleeping"
# start = time.time()

# #first do the gputhreads
# for j in range(NUM_CHUNKS):
#     for i in range(num_gpu_threads):
#         print "sending...", j, i, len(src_data)
#         inputpipein[i].send_bytes(src_data)
# #        time.sleep(.1)


# for i in range(num_gpu_threads):
#     print "receiving...", j, i
 
#     cpudata[i] = datapipein[i].recv()
# #        buf = datapipein[i].recv_bytes()
# #        print len(buf)
# #        buf2  = buf[1000:1000+4*CHUNK_LENGTH]
# #        cpudata[i] = struct.unpack("f"*int(len(buf2)/4), buf2)
#     print len(cpudata[i]), type(cpudata[i])

# print "waiting...."
# for ct,gt in zip(cpu_threads,gpu_threads):
#     ct.join()
#     gt.join()

# stop = time.time()
# cpu_time = stop - start 
# print "CPU+GPU Elapsed time: %s" % (cpu_time)
# print " CPU+GPU exerpt:", list(cpudata[0][i] for i in range(300,304))


ct = all_CPU_Thread(1, splice[0][2])
start = time.time()
ct.start()
ct.join()
stop = time.time() - start
print "all CPU Elapsed time: %s" % (stop)

gt = all_GPU_Thread(2, splice[0][2])
start = time.time()
gt.start()
gt.join()
stop = time.time() - start
print "all GPU Elapsed time: %s" % (stop)


# #now do the cpu only control
# start = time.time()
# for ct,ct2 in zip(cpu1_threads,cpu2_threads):
#     ct.start()
#     ct2.start()

# i = num_threads - 1
# for j in range(NUM_CHUNKS):
#     print "sending...", j, i
#     inputpipein[i].send_bytes(src_data)

# print "recieving...", i
# cpudata1 = datapipein[i].recv()
# print "done"

# for ct,ct2 in zip(cpu1_threads,cpu2_threads):
#     ct.join()
#     ct2.join()

# stop = time.time()
# cpu_time = stop - start 
# print "CPU+CPU Elapsed time: %s" % (cpu_time)

#print " CPU+CPU exerpt:", list(cpudata[num_threads-1][i] for i in range(300,304))



if(False):
    s = 300
    l = 50
    x = range(l)
    srcplot = list(src_data[i] for i in range(s, s+l)) 
    cpuplot = list(cpudata[0][i] for i in range(s, s+l)) 
    plot(x, srcplot, 'b-', x, cpuplot, 'g-')
    grid(True)
    show()


