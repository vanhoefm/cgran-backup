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
from openrd import pr

## @package dmod
# dmod module containing classes for (de)modulating data

## Signal shaping using root-raised cosine filter
#
# \param mod_type modulation type (currently only BPSK)
# \param samples_per_symbol number of samples per symbol
class transmitter_cc(gr.hier_block2):
	def __init__(self, mod_type, samples_per_symbol):
		gr.hier_block2.__init__(self, "transmitter_cc",
				gr.io_signature(1, 1, gr.sizeof_gr_complex),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))

		# Filter bandwidth parameter
		self.d_excess_bw = 0.35

		# Create filter
		rrcntaps = 11*samples_per_symbol
		rrctaps = gr.firdes.root_raised_cosine(samples_per_symbol,
				samples_per_symbol, 1.0, self.d_excess_bw, rrcntaps)
		self.b_rrc = gr.interp_fir_filter_ccf(samples_per_symbol, rrctaps)

		self.connect(self, self.b_rrc, self)

class receiver_cc(gr.hier_block2):
	def __init__(self, mod_type, samples_per_symbol):
		gr.hier_block2.__init__(self, "receiver_cc",
				gr.io_signature(1, 1, gr.sizeof_gr_complex),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))

		self.d_mod_type = mod_type
		self.d_samples_per_symbol = samples_per_symbol
		
		# RRC filter parameters
		self.d_excess_bw = 0.35

		# Costas loop parameters
		self.d_costas_alpha = 0.1
		self.d_costas_beta = 0.25 * self.d_costas_alpha * self.d_costas_alpha
		self.d_costas_fmin = -0.1
		self.d_costas_fmax = 0.1

		# Muller and Mueller parameters
		self.d_mm_mu = 0.5
		self.d_mm_gain_mu = 0.01
		self.d_mm_omega = self.d_samples_per_symbol
		self.d_mm_gain_omega = 0.25 * self.d_mm_gain_mu * self.d_mm_gain_mu
		self.d_mm_omega_rel = 0.005

		chan_coeffs = gr.firdes.low_pass(1.0, samples_per_symbol, 1.2, 0.5, gr.firdes.WIN_HANN)
		self.b_channel_filter = gr.fft_filter_ccc(1, chan_coeffs)

		# Right now, only BPSK supported. Quit if other values are used.
		if self.d_mod_type == pr.MODULATION_QPSK:
			print "no qpsk yet"
			sys.exit(1)
		elif self.d_mod_type != pr.MODULATION_BPSK:
			raise ValueError("Invalid modulation type")

		# Create AGC
		self.b_agc = gr.agc_cc(rate=1e-3, max_gain=1e4)

		# Create rrc filter
		rrcntaps = 11*self.d_samples_per_symbol
		rrctaps = gr.firdes.root_raised_cosine(1.0, self.d_samples_per_symbol, 1.0, self.d_excess_bw, rrcntaps)
		self.b_rrc = gr.interp_fir_filter_ccf(1, rrctaps)

		# Phase, frequency, and symbol synchronization
		self.b_receiver = gr.mpsk_receiver_cc(2, 0, 
				self.d_costas_alpha, self.d_costas_beta,
				self.d_costas_fmin, self.d_costas_fmax,
				self.d_mm_mu, self.d_mm_gain_mu,
				self.d_mm_omega, self.d_mm_gain_omega, self.d_mm_omega_rel)

		# Connect the flow graph
		self.connect(self, self.b_channel_filter, self.b_agc, self.b_rrc, self.b_receiver, self)

class modulator_bc(gr.hier_block2):
	def __init__(self, mod_type):
		gr.hier_block2.__init__(self, "modulator_bc", 
				gr.io_signature(1, 1, gr.sizeof_char),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))
		
		self.d_mod_type = mod_type

		# Right now, only BPSK supported. Quit if other values are used.
		if self.d_mod_type == pr.MODULATION_QPSK:
			print "no qpsk yet"
			sys.exit(1)
		elif self.d_mod_type != pr.MODULATION_BPSK:
			raise ValueError("Invalid modulation type")
		
		# Create differential encoder
		self.b_diffenc = gr.diff_encoder_bb(2)

		# Symbol mapping
		bpskconst = [-1, 1]
		self.b_chunks2symbols = gr.chunks_to_symbols_bc(bpskconst)

		# Connect the flow graph
		self.connect(self, self.b_diffenc, self.b_chunks2symbols, self)

	def mod_type(self):
		return self.d_mod_type

class demodulator_cc(gr.hier_block2):
	def __init__(self, mod_type):
		gr.hier_block2.__init__(self, "demodulator_cc",
				gr.io_signature(1, 1, gr.sizeof_gr_complex),
				gr.io_signature(1, 1, gr.sizeof_gr_complex))

		self.d_mod_type = mod_type

		# Right now, only BPSK supported. Quit if other values are used.
		if self.d_mod_type == pr.MODULATION_QPSK:
			print "no qpsk yet"
			sys.exit(1)
		elif self.d_mod_type != pr.MODULATION_BPSK:
			raise ValueError("Invalid modulation type")
		
		# Differential decoding
		self.b_phasor = gr.diff_phasor_cc()
		
		# Reverse the differential decoding convention
		self.b_diffcorr = gr.multiply_const_cc(-1)

		# Connect the flow graph
		self.connect(self, self.b_phasor, self.b_diffcorr, self)

	def mod_type(self):
		return self.d_mod_type

