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
from openrd import pr, dmod, pyblk

class source_aaf(pyblk.source_node):
	def __init__(self, block_size, coding, framing, frame_size, modulation, samples_per_symbol):
		pyblk.source_node.__init__(self, "source_aaf", block_size)

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
			self.frame = pr.framer_simple_vbb(frame_size, pyblk.simple_access_code)
		elif framing == pr.FRAMING_GSM:
			self.frame = pr.framer_gsm_vbb(frame_size, pr.FIELD_CODE_R2_RM_11_78, pyblk.gsm_sync_code, pyblk.gsm_data_code)
		else:
			raise ValueError("Invalid framing type")

		# Create partitioner, modulator and transmitter
		self.part = pr.block_partition_vbb(self.coder.codeword_size(), self.frame.data_size())
		self.v2s = pr.pvec_to_stream(gr.sizeof_char, self.frame.frame_size())
		self.mod = dmod.modulator_bc(modulation)
		self.xmit = dmod.transmitter_cc(modulation, samples_per_symbol)

		self.tempext = pr.pvec_extract(pr.pvec_alloc_size(16+frame_size), 16, padvec.alloc_size(frame_size))

		# Connect the flow graph
		self.connect(self, self.coder, self.part, self.frame, self.tempext, self.v2s, self.mod, self.xmit, self)

class relay_aaf(pyblk.relay_node):
	def __init__(self, mode='fixed', gain=1.0):
		pyblk.relay_node.__init__(self, "relay_aaf")

		if mode=='fixed':
			self.scale = gr.multiply_const_cc(gain)
		else:
			self.scale = gr.agc_cc(rate=0.01, reference=gain)

		self.connect(self, self.scale, self)

class dest_aaf(pyblk.dest_node):
	def __init__(self, block_size, coding, framing, frame_size, modulation):
		pyblk.dest_node.__init__(self, "dest_aaf", block_size)

		self.d_coding = coding
		self.d_framing = framing
		self.d_frame_size = frame_size
		self.d_modulation = modulation

		# Number of symbols in a frame
		frame_symbols = frame_size/pr.bits_per_symbol(modulation)

		# Receiver, demodulator and constellation decoder
		self.s1demod = dmod.demodulator_cc(modulation)
		self.s2demod = dmod.demodulator_cc(modulation)
		self.s1constdec = pr.constellation_decoder_cb(modulation)
		self.s2constdec = pr.constellation_decoder_cb(modulation)

		# Create frame correlators and deframer blocks
		if framing == pr.FRAMING_NONE:
			self.s1framecorr = pr.frame_correlator_none_bb(self.s1constdec.symbol_bits(), frame_symbols)
			self.s2framecorr = pr.frame_correlator_none_bb(self.s2constdec.symbol_bits(), frame_symbols)
			self.s1deframe = pr.deframer_none_vcc(frame_symbols)
			self.s2deframe = pr.deframer_none_vcc(frame_symbols)
		elif framing == pr.FRAMING_SIMPLE:
			self.s1framecorr = pr.frame_correlator_simple_bb(self.s1constdec.symbol_bits(), frame_symbols, pyblk.simple_access_code, 48)
			self.s2framecorr = pr.frame_correlator_simple_bb(self.s2constdec.symbol_bits(), frame_symbols, pyblk.simple_access_code, 48)
			self.s1deframe = pr.deframer_simple_vcc(frame_symbols, pyblk.simple_access_code)
			self.s2deframe = pr.deframer_simple_vcc(frame_symbols, pyblk.simple_access_code)
		elif framing == pr.FRAMING_GSM:
			self.s1framecorr = pr.frame_correlator_gsm_bb(self.s1constdec.symbol_bits(), frame_symbols, pyblk.gsm_sync_code, 60, pyblk.gsm_data_code, 24)
			self.s2framecorr = pr.frame_correlator_gsm_bb(self.s2constdec.symbol_bits(), frame_symbols, pyblk.gsm_sync_code, 60, pyblk.gsm_data_code, 24)
			self.s1deframe = pr.deframer_gsm_vcc(frame_symbols, pr.FIELD_CODE_R2_RM_11_78, pyblk.gsm_sync_code, pyblk.gsm_data_code)
			self.s2deframe = pr.deframer_gsm_vcc(frame_symbols, pr.FIELD_CODE_R2_RM_11_78, pyblk.gsm_sync_code, pyblk.gsm_data_code)
		else:
			raise ValueError("Invalid framing type")

		# Delay the inputs by the access code length
		self.s1d = gr.delay(gr.sizeof_gr_complex, self.s1framecorr.delay()-1)
		self.s2d = gr.delay(gr.sizeof_gr_complex, self.s2framecorr.delay()-1)

		# Create frame vectors from symbol streams
		self.s1fsync = pr.frame_sync_cc(frame_symbols)
		self.s2fsync = pr.frame_sync_cc(frame_symbols)

		# Number of symbols and bits in a partition
		part_symbols = self.s1deframe.data_size()
		part_size = part_symbols*pr.bits_per_symbol(modulation)

		# Create packet synchronizer and MRC
		self.psync = pr.packet_sync_vcc(part_symbols, pr.SEQPOLICY_SYNC, 2000)
		self.mrc = pr.mrc_vcc(part_symbols)

		# Soft constellation point parsing on merged vector
		self.constparse = pr.constellation_softdecoder_vcf(modulation, part_symbols)

		# Decoder
		if coding == pr.CODING_NONE:
			self.decoder = pr.block_decoder_none_vfb(block_size)
		else:
			raise ValueError("Invalid coding type")

		# Merge partitions to a packet that can be sent to the decoder
		self.merge = pr.block_merge_vff(part_symbols, self.decoder.codeword_size())

		# Set number of frames per packet
		frames_per_packet = self.decoder.codeword_size()/(self.s1deframe.data_size()*pr.bits_per_symbol(modulation))
		if framing == pr.FRAMING_GSM:
			self.s1framecorr.set_maxdataframes(frames_per_packet)
			self.s2framecorr.set_maxdataframes(frames_per_packet)

		# Connect the flow graph
		self.connect((self, 0), self.s1demod, self.s1d, (self.s1fsync, 0))
		self.connect((self, 1), self.s2demod, self.s2d, (self.s2fsync, 0))
		self.connect(self.s1demod, self.s1constdec, self.s1framecorr, (self.s1fsync, 1))
		self.connect(self.s2demod, self.s2constdec, self.s2framecorr, (self.s2fsync, 1))
		self.connect(self.s1fsync, self.s1deframe, (self.psync, 0), (self.mrc, 0))
		self.connect(self.s2fsync, self.s2deframe, (self.psync, 1), (self.mrc, 1))
		self.connect(self.mrc, self.constparse, self.merge, self.decoder, self)

