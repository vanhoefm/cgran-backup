#!/usr/bin/env python
#
# Copyright 2011 Anton Blad.
# 
# This file is part of OpenRD
# 
# OpenRD is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# OpenRD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from gnuradio import gr, gr_unittest
from openrd import pr, qa
from threading import Thread

class source(Thread):
	def __init__(self, psrc, proto, length, numpkt):
		Thread.__init__(self)

		self.psrc = psrc
		self.proto = proto
		self.length = length
		self.numpkt = numpkt

	def run(self):
		self.data = {}
		for k in range(self.numpkt):
			data = range(self.length)
			data[0] = self.proto
			data[1] = k%256
			self.psrc.send_data(self.proto, data)
			self.data[k] = data

class sink(Thread):
	def __init__(self, psink, proto, length, numpkt):
		Thread.__init__(self)

		self.psink = psink
		self.proto = proto
		self.length = length
		self.numpkt = numpkt

	def run(self):
		self.data = {}
		for k in range(self.numpkt):
			self.data[k] = self.psink.recv_data(self.proto)

class qa_pr_packet_endpoints(gr_unittest.TestCase):
	def setUp(self):
		self.tb = gr.top_block()
	
	def tearDown(self):
		self.tb = None
	
	def test_001_packet_endpoints(self):
		num_proto = 2
		data_size = 16
		block_size = 48+8*data_size
		numpkt = 100

		# Instantiate units under test
		uut1 = pr.data_source_packet(num_proto, block_size)
		uut2 = pr.data_sink_packet(num_proto, block_size)

		# Fiddle with the header to create rxmeta from txmeta
		eseq = pr.pvec_extract(pr.sizeof_txmeta + block_size, 0, 4)
		edata = pr.pvec_extract(pr.sizeof_txmeta + block_size, pr.sizeof_txmeta, block_size)
		ghead = gr.vector_source_i([0], True)
		bm = pr.pvec_concat([4, 4, block_size])

		src0 = source(uut1, 0, 32, numpkt)
		src1 = source(uut1, 1, 38, numpkt)
		sink0 = sink(uut2, 0, 32, numpkt)
		sink1 = sink(uut2, 1, 38, numpkt)

		self.tb.connect(uut1, eseq)
		self.tb.connect(uut1, edata)
		self.tb.connect(eseq, (bm, 0))
		self.tb.connect(ghead, (bm, 1))
		self.tb.connect(edata, (bm, 2))
		self.tb.connect(bm, uut2)

		# Start the flow graph
		self.tb.start()

		# Start the source and sink threads
		src0.start()
		src1.start()
		sink0.start()
		sink1.start()

		# Wait for all threads to finish
		src0.join()
		src1.join()
		sink0.join()
		sink1.join()

		# Stop the flow graph
		self.tb.stop()

		# Check the received data
		for k in range(numpkt):
			self.assertFloatTuplesAlmostEqual(src0.data[k], sink0.data[k])
			self.assertFloatTuplesAlmostEqual(src1.data[k], sink1.data[k])

if __name__ == '__main__':
	pass
	#gr_unittest.main()

