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

from gnuradio import gr
from openrd import pr

def data_size(dtype):
	if dtype == "b":
		return gr.sizeof_char
	elif dtype == "f":
		return gr.sizeof_float
	elif dtype == "c":
		return gr.sizeof_gr_complex

class pvec_source_b(gr.hier_block2):
	def __init__(self, data, vlen):
		gr.hier_block2.__init__(self, "pvec_source_b",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_char)))

		src = gr.vector_source_b(data)
		s2v = pr.stream_to_pvec(gr.sizeof_char, vlen)

		self.connect(src, s2v, self)

class pvec_source_c(gr.hier_block2):
	def __init__(self, data, vlen):
		gr.hier_block2.__init__(self, "pvec_source_c",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_gr_complex)))

		src = gr.vector_source_c(data)
		s2v = pr.stream_to_pvec(gr.sizeof_gr_complex, vlen)

		self.connect(src, s2v, self)

class pvec_source_f(gr.hier_block2):
	def __init__(self, data, vlen):
		gr.hier_block2.__init__(self, "pvec_source_f",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_float)))

		src = gr.vector_source_f(data)
		s2v = pr.stream_to_pvec(gr.sizeof_float, vlen)

		self.connect(src, s2v, self)

class pvec_source_i(gr.hier_block2):
	def __init__(self, data, vlen):
		gr.hier_block2.__init__(self, "pvec_source_i",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_int)))

		src = gr.vector_source_i(data)
		s2v = pr.stream_to_pvec(gr.sizeof_int, vlen)

		self.connect(src, s2v, self)

class pvec_source_s(gr.hier_block2):
	def __init__(self, data, vlen):
		gr.hier_block2.__init__(self, "pvec_source_s",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_short)))

		src = gr.vector_source_s(data)
		s2v = pr.stream_to_pvec(gr.sizeof_short, vlen)

		self.connect(src, s2v, self)

class pvec_sink_b(gr.hier_block2):
	def __init__(self, vlen):
		gr.hier_block2.__init__(self, "pvec_sink_b",
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_char)),
				gr.io_signature(0, 0, 0))

		v2s = pr.pvec_to_stream(gr.sizeof_char, vlen)
		self.dest = gr.vector_sink_b()

		self.connect(self, v2s, self.dest)

	def data(self):
		return self.dest.data()

class pvec_sink_c(gr.hier_block2):
	def __init__(self, vlen):
		gr.hier_block2.__init__(self, "pvec_sink_c",
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_gr_complex)),
				gr.io_signature(0, 0, 0))

		v2s = pr.pvec_to_stream(gr.sizeof_gr_complex, vlen)
		self.dest = gr.vector_sink_c()

		self.connect(self, v2s, self.dest)

	def data(self):
		return self.dest.data()

class pvec_sink_f(gr.hier_block2):
	def __init__(self, vlen):
		gr.hier_block2.__init__(self, "pvec_sink_f",
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_float)),
				gr.io_signature(0, 0, 0))

		v2s = pr.pvec_to_stream(gr.sizeof_float, vlen)
		self.dest = gr.vector_sink_f()

		self.connect(self, v2s, self.dest)

	def data(self):
		return self.dest.data()

class pvec_sink_i(gr.hier_block2):
	def __init__(self, vlen):
		gr.hier_block2.__init__(self, "pvec_sink_i",
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_int)),
				gr.io_signature(0, 0, 0))

		v2s = pr.pvec_to_stream(gr.sizeof_int, vlen)
		self.dest = gr.vector_sink_i()

		self.connect(self, v2s, self.dest)

	def data(self):
		return self.dest.data()

class pvec_sink_s(gr.hier_block2):
	def __init__(self, vlen):
		gr.hier_block2.__init__(self, "pvec_sink_s",
				gr.io_signature(1, 1, pr.pvec_alloc_size(vlen*gr.sizeof_short)),
				gr.io_signature(0, 0, 0))

		v2s = pr.pvec_to_stream(gr.sizeof_short, vlen)
		self.dest = gr.vector_sink_s()

		self.connect(self, v2s, self.dest)

	def data(self):
		return self.dest.data()

def pvec_source_x(dtype, data, vlen):
	if dtype == "b":
		return pvec_source_b(data, vlen)
	elif dtype == "f":
		return pvec_source_f(data, vlen)
	elif dtype == "c":
		return pvec_source_c(data, vlen)

def pvec_sink_x(dtype, vlen):
	if dtype == "b":
		return pvec_sink_b(vlen)
	elif dtype == "f":
		return pvec_sink_f(vlen)
	elif dtype == "c":
		return pvec_sink_c(vlen)

class rxframe_source(gr.hier_block2):
	def __init__(self, power, stamp, pkt_seq, frame_seq, frame_type):
		gr.hier_block2.__init__(self, "rxframe_source",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_rxframe)))

		bsrc0 = gr.vector_source_f(power)
		bsrc1 = gr.vector_source_i(stamp)
		bsrc2 = gr.vector_source_i(pkt_seq)
		bsrc3 = gr.vector_source_i(frame_seq)
		bsrc4 = gr.vector_source_i(frame_type)
		bsrc5 = gr.vector_source_i([0], repeat=True)
		bm = pr.pvec_concat([4, 4, 4, 4, 4, 4])

		self.connect(bsrc0, (bm, 0))
		self.connect(bsrc1, (bm, 1))
		self.connect(bsrc2, (bm, 2))
		self.connect(bsrc3, (bm, 3))
		self.connect(bsrc4, (bm, 4))
		self.connect(bsrc5, (bm, 5))
		self.connect(bm, self)

class rxframe_sink(gr.hier_block2):
	def __init__(self):
		block_size = pr.pvec_alloc_size(pr.sizeof_rxframe)

		gr.hier_block2.__init__(self, "rxframe_sink",
				gr.io_signature(1, 1, block_size),
				gr.io_signature(0, 0, 0))

		self.bsink0 = gr.vector_sink_f()
		self.bsink1 = gr.vector_sink_i()
		self.bsink2 = gr.vector_sink_i()
		self.bsink3 = gr.vector_sink_i()
		self.bsink4 = gr.vector_sink_i()
		be0 = pr.pvec_extract(block_size, 0, 4)
		be1 = pr.pvec_extract(block_size, 4, 4)
		be2 = pr.pvec_extract(block_size, 8, 4)
		be3 = pr.pvec_extract(block_size, 12, 4)
		be4 = pr.pvec_extract(block_size, 16, 4)

		self.connect(self, be0, self.bsink0)
		self.connect(self, be1, self.bsink1)
		self.connect(self, be2, self.bsink2)
		self.connect(self, be3, self.bsink3)
		self.connect(self, be4, self.bsink4)

	def power(self):
		return self.bsink0.data()

	def stamp(self):
		return self.bsink1.data()

	def pkt_seq(self):
		return self.bsink2.data()

	def frame_seq(self):
		return self.bsink3.data()

	def frame_type(self):
		return self.bsink4.data()

class rxmeta_source(gr.hier_block2):
	def __init__(self, pkt_seq, decoded):
		gr.hier_block2.__init__(self, "rxmeta_source",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_rxmeta)))

		bsrc0 = gr.vector_source_i(pkt_seq)
		bsrc1 = gr.vector_source_b(decoded)
		bsrc2 = gr.vector_source_b([0], repeat=True)
		bsrc3 = gr.vector_source_s([0], repeat=True)
		bm = pr.pvec_concat([4, 1, 1, 2])

		self.connect(bsrc0, (bm, 0))
		self.connect(bsrc1, (bm, 1))
		self.connect(bsrc2, (bm, 2))
		self.connect(bsrc3, (bm, 3))
		self.connect(bm, self)

class rxmeta_sink(gr.hier_block2):
	def __init__(self):
		block_size = pr.pvec_alloc_size(pr.sizeof_rxmeta)

		gr.hier_block2.__init__(self, "rxmeta_sink",
				gr.io_signature(1, 1, block_size),
				gr.io_signature(0, 0, 0))

		self.bsink0 = gr.vector_sink_i()
		self.bsink1 = gr.vector_sink_b()
		be0 = pr.pvec_extract(block_size, 0, 4)
		be1 = pr.pvec_extract(block_size, 4, 1)

		self.connect(self, be0, self.bsink0)
		self.connect(self, be1, self.bsink1)

	def pkt_seq(self):
		return self.bsink0.data()

	def decoded(self):
		return self.bsink1.data()

class txmeta_source(gr.hier_block2):
	def __init__(self, pkt_seq, frame_seq, data_valid):
		gr.hier_block2.__init__(self, "txmeta_source",
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_txmeta)))

		bsrc0 = gr.vector_source_i(pkt_seq)
		bsrc1 = gr.vector_source_i(frame_seq)
		bsrc2 = gr.vector_source_b(data_valid)
		bsrc3 = gr.vector_source_b([0], repeat=True)
		bsrc4 = gr.vector_source_s([0], repeat=True)
		bsrc5 = gr.vector_source_i([0], repeat=True)
		bm = pr.pvec_concat([4, 4, 1, 1, 2, 4])

		self.connect(bsrc0, (bm, 0))
		self.connect(bsrc1, (bm, 1))
		self.connect(bsrc2, (bm, 2))
		self.connect(bsrc3, (bm, 3))
		self.connect(bsrc4, (bm, 4))
		self.connect(bsrc5, (bm, 5))
		self.connect(bm, self)

class txmeta_sink(gr.hier_block2):
	def __init__(self):
		block_size = pr.pvec_alloc_size(pr.sizeof_txmeta)

		gr.hier_block2.__init__(self, "txmeta_sink",
				gr.io_signature(1, 1, block_size),
				gr.io_signature(0, 0, 0))

		self.bsink0 = gr.vector_sink_i()
		self.bsink1 = gr.vector_sink_i()
		self.bsink2 = gr.vector_sink_b()
		be0 = pr.pvec_extract(block_size, 0, 4)
		be1 = pr.pvec_extract(block_size, 4, 4)
		be2 = pr.pvec_extract(block_size, 8, 1)

		self.connect(self, be0, self.bsink0)
		self.connect(self, be1, self.bsink1)
		self.connect(self, be2, self.bsink2)

	def pkt_seq(self):
		return self.bsink0.data()

	def frame_seq(self):
		return self.bsink1.data()

	def data_valid(self):
		return self.bsink2.data()

class rxframe_source_x(gr.hier_block2):
	def __init__(self, dtype, power, stamp, pkt_seq, frame_seq, frame_type, data, vlen):
		dsize = data_size(dtype)

		block_size = pr.pvec_alloc_size(pr.sizeof_rxframe+vlen*dsize)

		gr.hier_block2.__init__(self, "rxframe_source_" + dtype,
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, block_size))

		hsrc = rxframe_source(power, stamp, pkt_seq, frame_seq, frame_type)
		dsrc = pvec_source_x(dtype, data, vlen)
		bm = pr.pvec_concat([pr.sizeof_rxframe, vlen*dsize])

		self.connect(hsrc, (bm, 0))
		self.connect(dsrc, (bm, 1))
		self.connect(bm, self)

class rxframe_sink_x(gr.hier_block2):
	def __init__(self, dtype, vlen):
		dsize = data_size(dtype)

		block_size = pr.pvec_alloc_size(pr.sizeof_rxframe+vlen*dsize)

		gr.hier_block2.__init__(self, "rxframe_sink_" + dtype,
				gr.io_signature(1, 1, block_size),
				gr.io_signature(0, 0, 0))

		self.bs0 = rxframe_sink()
		self.bs1 = pvec_sink_x(dtype, vlen)

		be0 = pr.pvec_extract(block_size, 0, pr.sizeof_rxframe)
		be1 = pr.pvec_extract(block_size, pr.sizeof_rxframe, vlen*dsize)

		self.connect(self, be0, self.bs0)
		self.connect(self, be1, self.bs1)

	def power(self):
		return self.bs0.power()

	def stamp(self):
		return self.bs0.stamp()

	def pkt_seq(self):
		return self.bs0.pkt_seq()

	def frame_seq(self):
		return self.bs0.frame_seq()

	def frame_type(self):
		return self.bs0.frame_type()

	def data(self):
		return self.bs1.data()

class rxmeta_source_x(gr.hier_block2):
	def __init__(self, dtype, pkt_seq, decoded, data, vlen):
		dsize = data_size(dtype)

		block_size = pr.pvec_alloc_size(pr.sizeof_rxmeta+vlen*dsize)

		gr.hier_block2.__init__(self, "rxmeta_source_" + dtype,
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, block_size))

		hsrc = rxmeta_source(pkt_seq, decoded)
		dsrc = pvec_source_x(dtype, data, vlen)
		bm = pr.pvec_concat([pr.sizeof_rxmeta, vlen*dsize])

		self.connect(hsrc, (bm, 0))
		self.connect(dsrc, (bm, 1))
		self.connect(bm, self)

class rxmeta_sink_x(gr.hier_block2):
	def __init__(self, dtype, vlen):
		dsize = data_size(dtype)

		block_size = pr.pvec_alloc_size(pr.sizeof_rxmeta+vlen*dsize)

		gr.hier_block2.__init__(self, "rxmeta_sink_" + dtype,
				gr.io_signature(1, 1, block_size),
				gr.io_signature(0, 0, 0))

		self.bs0 = rxmeta_sink()
		self.bs1 = pvec_sink_x(dtype, vlen)

		be0 = pr.pvec_extract(pr.sizeof_rxmeta+vlen*dsize, 0, pr.sizeof_rxmeta)
		be1 = pr.pvec_extract(pr.sizeof_rxmeta+vlen*dsize, pr.sizeof_rxmeta, vlen*dsize)

		self.connect(self, be0, self.bs0)
		self.connect(self, be1, self.bs1)

	def pkt_seq(self):
		return self.bs0.pkt_seq()

	def decoded(self):
		return self.bs0.decoded()

	def data(self):
		return self.bs1.data()

class txmeta_source_x(gr.hier_block2):
	def __init__(self, dtype, pkt_seq, frame_seq, data_valid, data, vlen):
		dsize = data_size(dtype)

		block_size = pr.pvec_alloc_size(pr.sizeof_txmeta+vlen*dsize)

		gr.hier_block2.__init__(self, "txmeta_source_" + dtype,
				gr.io_signature(0, 0, 0),
				gr.io_signature(1, 1, block_size))

		hsrc = txmeta_source(pkt_seq, frame_seq, data_valid)
		dsrc = pvec_source_x(dtype, data, vlen)
		bm = pr.pvec_concat([pr.sizeof_txmeta, vlen*dsize])

		self.connect(hsrc, (bm, 0))
		self.connect(dsrc, (bm, 1))
		self.connect(bm, self)

class txmeta_sink_x(gr.hier_block2):
	def __init__(self, dtype, vlen):
		dsize = data_size(dtype)

		block_size = pr.pvec_alloc_size(pr.sizeof_txmeta+vlen*dsize)

		gr.hier_block2.__init__(self, "txmeta_sink_" + dtype,
				gr.io_signature(1, 1, block_size),
				gr.io_signature(0, 0, 0))

		self.bs0 = txmeta_sink()
		self.bs1 = pvec_sink_x(dtype, vlen)

		be0 = pr.pvec_extract(pr.sizeof_txmeta+vlen*dsize, 0, pr.sizeof_txmeta)
		be1 = pr.pvec_extract(pr.sizeof_txmeta+vlen*dsize, pr.sizeof_txmeta, vlen*dsize)

		self.connect(self, be0, self.bs0)
		self.connect(self, be1, self.bs1)

	def pkt_seq(self):
		return self.bs0.pkt_seq()

	def frame_seq(self):
		return self.bs0.frame_seq()

	def data_valid(self):
		return self.bs0.data_valid()

	def data(self):
		return self.bs1.data()

class rxframe_source_b(rxframe_source_x):
	def __init__(self, power, stamp, pkt_seq, frame_seq, frame_type, data, vlen):
		rxframe_source_x.__init__(self, "b", power, stamp, pkt_seq, frame_seq, frame_type, data, vlen)

class rxframe_sink_b(rxframe_sink_x):
	def __init__(self, vlen):
		rxframe_sink_x.__init__(self, "b", vlen)

class rxframe_source_f(rxframe_source_x):
	def __init__(self, power, stamp, pkt_seq, frame_seq, frame_type, data, vlen):
		rxframe_source_x.__init__(self, "f", power, stamp, pkt_seq, frame_seq, frame_type, data, vlen)

class rxframe_sink_f(rxframe_sink_x):
	def __init__(self, vlen):
		rxframe_sink_x.__init__(self, "f", vlen)

class rxframe_source_c(rxframe_source_x):
	def __init__(self, power, stamp, pkt_seq, frame_seq, frame_type, data, vlen):
		rxframe_source_x.__init__(self, "c", power, stamp, pkt_seq, frame_seq, frame_type, data, vlen)

class rxframe_sink_c(rxframe_sink_x):
	def __init__(self, vlen):
		rxframe_sink_x.__init__(self, "c", vlen)

class rxmeta_source_b(rxmeta_source_x):
	def __init__(self, pkt_seq, decoded, data, vlen):
		rxmeta_source_x.__init__(self, "b", pkt_seq, decoded, data, vlen)

class rxmeta_sink_b(rxmeta_sink_x):
	def __init__(self, vlen):
		rxmeta_sink_x.__init__(self, "b", vlen)

class rxmeta_source_f(rxmeta_source_x):
	def __init__(self, pkt_seq, decoded, data, vlen):
		rxmeta_source_x.__init__(self, "f", pkt_seq, decoded, data, vlen)

class rxmeta_sink_f(rxmeta_sink_x):
	def __init__(self, vlen):
		rxmeta_sink_x.__init__(self, "f", vlen)

class rxmeta_source_c(rxmeta_source_x):
	def __init__(self, pkt_seq, decoded, data, vlen):
		rxmeta_source_x.__init__(self, "c", pkt_seq, decoded, data, vlen)

class rxmeta_sink_c(rxmeta_sink_x):
	def __init__(self, vlen):
		rxmeta_sink_x.__init__(self, "c", vlen)

class txmeta_source_b(txmeta_source_x):
	def __init__(self, pkt_seq, frame_seq, data_valid, data, vlen):
		txmeta_source_x.__init__(self, "b", pkt_seq, frame_seq, data_valid, data, vlen)

class txmeta_sink_b(txmeta_sink_x):
	def __init__(self, vlen):
		txmeta_sink_x.__init__(self, "b", vlen)

