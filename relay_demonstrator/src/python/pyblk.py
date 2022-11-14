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

from gnuradio import gr, blks2
from openrd import pr, dmod

simple_access_code = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

gsm_sync_code = [1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1]
gsm_data_code = [1,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1,1]

class source_node(gr.hier_block2):
	def __init__(self, name, block_size):
		gr.hier_block2.__init__(self, name,
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_txmeta+block_size)),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))
		self.d_block_size = block_size
	
	def block_size():
		return self.d_block_size

class dest_node(gr.hier_block2):
	def __init__(self, name, block_size):
		gr.hier_block2.__init__(self, name,
				gr.io_signature(2, 2, gr.sizeof_gr_complex),
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_rxmeta+block_size)))
		self.d_block_size = block_size

	def block_size():
		return self.d_block_size

class relay_node(gr.hier_block2):
	def __init__(self, name):
		gr.hier_block2.__init__(self, name,
				gr.io_signature(1, 1, gr.sizeof_gr_complex),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))

class source_ref(gr.hier_block2):
	def __init__(self, block_size, coding, framing, frame_size, modulation):
		gr.hier_block2.__init__(self, "source_ref",
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_txmeta+block_size)),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))

		self.d_block_size = block_size
		self.d_coding = coding
		self.d_framing = framing
		self.d_frame_size = frame_size
		self.d_modulation = modulation

		# Create coder
		if coding == pr.CODING_NONE:
			self.coder = pr.block_coder_none_vbb(block_size)
		else:
			raise ValueError("Invalid coding type")

		# Create framer
		if framing == pr.FRAMING_NONE:
			self.frame = pr.framer_none_vbb(self.coder.codeword_size())
		elif framing == pr.FRAMING_SIMPLE:
			self.frame = pr.framer_simple_vbb(frame_size, simple_access_code)
		elif framing == pr.FRAMING_GSM:
			self.frame = pr.framer_gsm_vbb(frame_size, pr.FIELD_CODE_R2_RM_11_78, gsm_sync_code, gsm_data_code)
		else:
			raise ValueError("Invalid framing type")

		# Create partitioner, modulator and transmitter
		self.part = pr.block_partition_vbb(self.coder.codeword_size(), self.frame.data_size())
		self.v2s = pr.pvec_to_stream(gr.sizeof_char, self.frame.frame_size())
		self.mod = dmod.modulator_bc(modulation)

		self.tempext = pr.pvec_extract(pr.pvec_alloc_size(16+frame_size), 16, pr.pvec_alloc_size(frame_size))

		# Connect the flow graph
		self.connect(self, self.coder, self.part, self.frame, self.tempext, self.v2s, self.mod, self)

class dest_ref(gr.hier_block2):
	def __init__(self, block_size, coding, framing, frame_size, modulation):
		gr.hier_block2.__init__(self, "dest_ref",
				gr.io_signature(1, 1, gr.sizeof_gr_complex),
				gr.io_signature(1, 1, pr.pvec_alloc_size(pr.sizeof_rxmeta+block_size)))

		self.d_block_size = block_size
		self.d_coding = coding
		self.d_framing = framing
		self.d_frame_size = frame_size
		self.d_modulation = modulation

		# Number of symbols in a frame
		frame_symbols = frame_size/pr.bits_per_symbol(modulation)

		# Receiver, demodulator and constellation decoder
		self.demod = dmod.demodulator_cc(modulation)
		self.constdec = pr.constellation_decoder_cb(modulation)

		# Create frame correlator and deframer blocks
		if framing == pr.FRAMING_NONE:
			self.framecorr = pr.frame_correlator_none_bb(self.constdec.symbol_bits(), frame_symbols)
			self.deframe = pr.deframer_none_vcc(frame_symbols)
		elif framing == pr.FRAMING_SIMPLE:
			self.framecorr = pr.frame_correlator_simple_bb(self.constdec.symbol_bits(), 
					frame_symbols, simple_access_code, 48)
			self.deframe = pr.deframer_simple_vcc(frame_symbols, simple_access_code)
		elif framing == pr.FRAMING_GSM:
			self.framecorr = pr.frame_correlator_gsm_bb(self.constdec.symbol_bits(), frame_symbols, gsm_sync_code, 60, gsm_data_code, 24)
			self.deframe = pr.deframer_gsm_vcc(frame_symbols, pr.FIELD_CODE_R2_RM_11_78, gsm_sync_code, gsm_data_code)
		else:
			raise ValueError("Invalid framing type")

		# Delay the input by the access code length
		self.delay = gr.delay(gr.sizeof_gr_complex, self.framecorr.delay()-1)

		# Create frame vector from symbol stream
		self.fsync = pr.frame_sync_cc(frame_symbols)

		# Number of symbols in a partition
		part_symbols = self.deframe.data_size()

		# Soft constellation point parsing
		self.constparse = pr.constellation_softdecoder_vcf(modulation, part_symbols)

		# Decoder
		if coding == pr.CODING_NONE:
			self.decoder = pr.block_decoder_none_vfb(block_size)
		else:
			raise ValueError("Invalid coding type")

		# Merge partitions to a packet that can be sent to the decoder
		self.merge = pr.block_merge_vff(part_symbols, self.decoder.codeword_size())

		# Set number of frames per packet
		frames_per_packet = self.decoder.codeword_size()/(self.deframe.data_size()*pr.bits_per_symbol(modulation))
		if framing == pr.FRAMING_GSM:
			self.framecorr.set_maxdataframes(frames_per_packet)

		# Connect the flow graph
		self.connect(self, self.demod, self.delay, (self.fsync, 0))
		self.connect(self.demod, self.constdec, self.framecorr, (self.fsync, 1))
		self.connect(self.fsync, self.deframe, self.constparse, self.merge, self.decoder, self)

def data_size(dtype):
	if dtype == "b":
		return gr.sizeof_char
	elif dtype == "f":
		return gr.sizeof_float
	elif dtype == "c":
		return gr.sizeof_gr_complex

class txmeta_decap_x(gr.hier_block2):
	def __init__(self, dtype, vlen):
		dsize = data_size(dtype)

		insize = pr.pvec_alloc_size(pr.sizeof_txmeta+vlen*dsize)
		outsize = pr.pvec_alloc_size(vlen*dsize)

		gr.hier_block2.__init__(self, "txmeta_decap_" + dtype,
				gr.io_signature(1, 1, insize),
				gr.io_signature(1, 1, outsize))

		self.be = pr.pvec_extract(insize, pr.sizeof_txmeta, outsize)
		self.connect(self, self.be, self)

class rxframe_decap_x(gr.hier_block2):
	def __init__(self, dtype, vlen):
		dsize = data_size(dtype)

		insize = pr.pvec_alloc_size(pr.sizeof_rxframe+vlen*dsize)
		outsize = pr.pvec_alloc_size(vlen*dsize)

		gr.hier_block2.__init__(self, "rxframe_decap_" + dtype,
				gr.io_signature(1, 1, insize),
				gr.io_signature(1, 1, outsize))

		self.be = pr.pvec_extract(insize, pr.sizeof_rxframe, outsize)
		self.connect(self, self.be, self)

class rxmeta_decap_x(gr.hier_block2):
	def __init__(self, dtype, vlen):
		dsize = data_size(dtype)

		insize = pr.pvec_alloc_size(pr.sizeof_rxmeta+vlen*dsize)
		outsize = pr.pvec_alloc_size(vlen*dsize)

		gr.hier_block2.__init__(self, "rxmeta_decap_" + dtype,
				gr.io_signature(1, 1, insize),
				gr.io_signature(1, 1, outsize))

		self.be = pr.pvec_extract(insize, pr.sizeof_rxmeta, outsize)
		self.connect(self, self.be, self)

class txmeta_decap_b(txmeta_decap_x):
	def __init__(self, vlen):
		txmeta_decap_x.__init__(self, "b", vlen)

class txmeta_decap_f(txmeta_decap_x):
	def __init__(self, vlen):
		txmeta_decap_x.__init__(self, "f", vlen)

class txmeta_decap_c(txmeta_decap_x):
	def __init__(self, vlen):
		txmeta_decap_x.__init__(self, "c", vlen)

class rxframe_decap_b(rxframe_decap_x):
	def __init__(self, vlen):
		rxframe_decap_x.__init__(self, "b", vlen)

class rxframe_decap_f(rxframe_decap_x):
	def __init__(self, vlen):
		rxframe_decap_x.__init__(self, "f", vlen)

class rxframe_decap_c(rxframe_decap_x):
	def __init__(self, vlen):
		rxframe_decap_x.__init__(self, "c", vlen)

class rxmeta_decap_b(rxmeta_decap_x):
	def __init__(self, vlen):
		rxmeta_decap_x.__init__(self, "b", vlen)

class rxmeta_decap_f(rxmeta_decap_x):
	def __init__(self, vlen):
		rxmeta_decap_x.__init__(self, "f", vlen)

class rxmeta_decap_c(rxmeta_decap_x):
	def __init__(self, vlen):
		rxmeta_decap_x.__init__(self, "c", vlen)

